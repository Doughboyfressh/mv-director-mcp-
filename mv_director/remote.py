"""Remote (phone-friendly) mode: workspace on the server, fetching media from share links,
download links for results, background jobs, and an HTTP app protected by a secret key."""
from __future__ import annotations

import hmac
import os
import re
import threading
import time
import traceback
import urllib.parse
import uuid
from pathlib import Path

from .media import MediaError

WORKDIR = Path(os.environ.get("MVD_WORKDIR", "~/mv-director-workspace")).expanduser().resolve()
ACCESS_KEY = os.environ.get("MVD_ACCESS_KEY", "")
PUBLIC_URL = os.environ.get("MVD_PUBLIC_URL", "").rstrip("/")
MAX_DOWNLOAD_GB = float(os.environ.get("MVD_MAX_DOWNLOAD_GB", "20"))
REMOTE = False  # set True by enable_remote()


def enable_remote() -> None:
    global REMOTE
    REMOTE = True
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)  # relative paths in every tool resolve inside the workspace
    os.environ["MVD_SANDBOX"] = str(WORKDIR)


# ------------------------------------------------------------------ links
def download_link(path: str | Path) -> str | None:
    if not (REMOTE and PUBLIC_URL and ACCESS_KEY):
        return None
    p = Path(path).resolve()
    try:
        rel = p.relative_to(WORKDIR)
    except ValueError:
        return None
    return f"{PUBLIC_URL}/{ACCESS_KEY}/files/{urllib.parse.quote(str(rel))}"


# ------------------------------------------------------------------ fetching
def _direct_url(url: str) -> tuple[str, str]:
    """Turn share links into direct-download URLs. Returns (url, source)."""
    u = url.strip()
    m = (re.search(r"drive\.google\.com/file/d/([\w-]{20,})", u)
         or re.search(r"[?&]id=([\w-]{20,})", u) if "google" in u else None)
    if m:
        fid = m.group(1)
        return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t", "google_drive"
    if "dropbox.com" in u:
        u = re.sub(r"([?&])dl=0", r"\1dl=1", u)
        if "dl=1" not in u and "raw=1" not in u:
            u += ("&" if "?" in u else "?") + "dl=1"
        return u, "dropbox"
    if "1drv.ms" in u or "onedrive.live.com" in u:
        return u.replace("redir?", "download?").replace("embed?", "download?"), "onedrive"
    return u, "direct"


def _filename_from(resp, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    name = urllib.parse.unquote(m.group(1)) if m else fallback
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip() or "download.bin"
    return name


def fetch(url: str, subfolder: str = "inbox", filename: str | None = None, progress: dict | None = None) -> dict:
    import urllib.error
    import urllib.request
    if progress is None:
        progress = getattr(_TLS, "progress", None)
    durl, source = _direct_url(url)
    dest_dir = (WORKDIR / subfolder).resolve()
    if WORKDIR not in dest_dir.parents and dest_dir != WORKDIR:
        raise MediaError("subfolder must stay inside the workspace")
    dest_dir.mkdir(parents=True, exist_ok=True)
    limit = MAX_DOWNLOAD_GB * 1e9
    t0 = time.time()
    req = urllib.request.Request(durl, headers={"User-Agent": "mv-director/0.2"})
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        raise MediaError(f"Download failed: HTTP {e.code}. For Google Drive, set sharing to 'Anyone with the "
                         "link'. For Dropbox, use a public share link.") from e
    except urllib.error.URLError as e:
        raise MediaError(f"Could not reach the link: {e.reason}") from e
    with r:
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            raise MediaError("The link returned a web page, not a file. The file is probably private: set sharing "
                             "to 'Anyone with the link' (Drive) or use a public share link, then retry.")
        name = filename or _filename_from(r, Path(urllib.parse.urlparse(durl).path).name or "download.bin")
        dest = dest_dir / name
        total = int(r.headers.get("content-length", 0) or 0)
        if total > limit:
            raise MediaError(f"File is {total / 1e9:.1f} GB, over the {MAX_DOWNLOAD_GB} GB limit (MVD_MAX_DOWNLOAD_GB)")
        got = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got > limit:
                    raise MediaError("Download exceeded the size limit")
                if progress is not None:
                    progress.update({"mb": round(got / 1e6, 1), "total_mb": round(total / 1e6, 1) if total else None,
                                     "pct": round(100 * got / total, 1) if total else None})
        tmp.rename(dest)
    return {"path": str(dest), "relative_path": str(dest.relative_to(WORKDIR)), "source": source,
            "size_mb": round(got / 1e6, 1), "seconds": round(time.time() - t0, 1)}


def list_files(subfolder: str = "") -> list[dict]:
    base = (WORKDIR / subfolder).resolve()
    out = []
    if not base.exists():
        return out
    for p in sorted(base.rglob("*")):
        if p.is_file() and not p.name.endswith(".part") and p.name != "mvd_project.json":
            out.append({"path": str(p.relative_to(WORKDIR)), "size_mb": round(p.stat().st_size / 1e6, 1),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                        "download_url": download_link(p)})
    return out


# ------------------------------------------------------------------ background jobs
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_TLS = threading.local()


def start_job(name: str, fn, kwargs: dict) -> dict:
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "tool": name, "status": "running", "started": time.strftime("%H:%M:%S"),
           "t0": time.time(), "progress": {}, "result": None, "error": None}
    def run():
        _TLS.progress = job["progress"]
        try:
            job["result"] = fn(**kwargs)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = f"{e}"
            job["trace"] = traceback.format_exc()[-1500:]
        job["elapsed_s"] = round(time.time() - job["t0"], 1)
    with _LOCK:
        JOBS[jid] = job
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid, "status": "running", "tool": name,
            "note": "Check with job_status(job_id). Renders of a full video can take several minutes."}


def job_status(jid: str | None = None) -> dict:
    def view(j):
        v = {k: j[k] for k in ("id", "tool", "status", "started", "progress", "result", "error") if j.get(k) not in (None, {})}
        v["elapsed_s"] = j.get("elapsed_s") or round(time.time() - j["t0"], 1)
        return v
    if jid:
        if jid not in JOBS:
            raise MediaError(f"No job {jid}")
        return view(JOBS[jid])
    return {"jobs": [view(j) for j in list(JOBS.values())[-20:]]}


# ------------------------------------------------------------------ HTTP app
def build_app(mcp_server, host: str):
    """Starlette app: /<KEY>/mcp (MCP endpoint), /<KEY>/files/... (downloads), /health."""
    from starlette.responses import FileResponse, JSONResponse, PlainTextResponse

    try:
        from mcp.server.transport_security import TransportSecuritySettings
        sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        app = mcp_server.streamable_http_app(host=host, transport_security=sec)
    except (ImportError, TypeError):
        app = mcp_server.streamable_http_app()

    key = ACCESS_KEY.encode()

    class KeyGate:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.inner(scope, receive, send)
            path = scope.get("path", "")
            if path in ("/health", "/"):
                return await PlainTextResponse("music-video-director ok")(scope, receive, send)
            parts = path.split("/", 2)  # ['', key, rest]
            if len(parts) < 3 or not hmac.compare_digest(parts[1].encode(), key):
                return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
            rest = "/" + parts[2]
            if rest.startswith("/files/"):
                rel = urllib.parse.unquote(rest[len("/files/"):])
                p = (WORKDIR / rel).resolve()
                if (WORKDIR not in p.parents) or not p.is_file():
                    return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
                return await FileResponse(p, filename=p.name)(scope, receive, send)
            scope = dict(scope)
            scope["path"] = rest
            scope["raw_path"] = rest.encode()
            return await self.inner(scope, receive, send)

    return KeyGate(app)
