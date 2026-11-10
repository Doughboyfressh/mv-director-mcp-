"""ffmpeg / ffprobe helpers shared by every tool."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get("MVD_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = os.environ.get("MVD_FFPROBE", shutil.which("ffprobe") or "ffprobe")


try:  # make our errors visible to the model (MCP SDK 2.x hides other exception text)
    from mcp.server.mcpserver.exceptions import ToolError as _Base
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError as _Base
    except ImportError:
        _Base = RuntimeError


class MediaError(_Base):
    pass


def check_binaries() -> dict:
    out = {}
    for name, b in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)):
        try:
            v = subprocess.run([b, "-version"], capture_output=True, text=True, timeout=10)
            out[name] = v.stdout.splitlines()[0] if v.returncode == 0 else "error"
        except FileNotFoundError:
            out[name] = "NOT FOUND"
    return out


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-15:])
        raise MediaError(f"Command failed ({Path(cmd[0]).name}):\n{tail}")
    return p


def ffmpeg(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return run([FFMPEG, "-hide_banner", "-nostdin", "-y", *args], timeout=timeout)


def resolve(path: str, must_exist: bool = True) -> Path:
    p = Path(os.path.expanduser(path)).resolve()
    box = os.environ.get("MVD_SANDBOX")
    if box and p != Path(box) and Path(box) not in p.parents:
        raise MediaError(f"Path {path} is outside the server workspace. Use paths relative to the workspace "
                         "(e.g. inbox/video.mp4); see list_workspace.")
    if must_exist and not p.exists():
        raise MediaError(f"File not found: {p}")
    return p


def out_path(path: str) -> Path:
    p = resolve(path, must_exist=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def probe(path: str | Path) -> dict:
    p = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format",
             "-show_streams", str(path)])
    return json.loads(p.stdout)


def _rate(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


def video_stream(info: dict) -> dict | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic"):
            return s
    return None


def audio_stream(info: dict) -> dict | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def duration(info: dict) -> float:
    d = info.get("format", {}).get("duration")
    if d:
        return float(d)
    for s in info.get("streams", []):
        if s.get("duration"):
            return float(s["duration"])
    return 0.0


def fps_of(info: dict) -> float | None:
    v = video_stream(info)
    if not v:
        return None
    return _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))


def summarize(path: str | Path) -> dict:
    """Videographer-oriented summary of a media file."""
    info = probe(path)
    v, a = video_stream(info), audio_stream(info)
    fmt = info.get("format", {})
    tags = {k.lower(): val for k, val in (fmt.get("tags") or {}).items()}
    out: dict = {
        "file": str(path),
        "container": fmt.get("format_name"),
        "duration_s": round(duration(info), 3),
        "size_mb": round(int(fmt.get("size", 0)) / 1e6, 1),
        "bitrate_mbps": round(int(fmt.get("bit_rate", 0) or 0) / 1e6, 2),
        "timecode": tags.get("timecode"),
        "camera": tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("encoder"),
        "warnings": [],
    }
    if v:
        r_fps, a_fps = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
        vtags = {k.lower(): val for k, val in (v.get("tags") or {}).items()}
        out["video"] = {
            "codec": v.get("codec_name"), "profile": v.get("profile"),
            "width": v.get("width"), "height": v.get("height"),
            "fps": round(a_fps or r_fps or 0, 3),
            "pix_fmt": v.get("pix_fmt"),
            "bit_depth": int(v.get("bits_per_raw_sample") or 0) or (10 if "10" in (v.get("pix_fmt") or "") else 8),
            "color_primaries": v.get("color_primaries"),
            "color_transfer": v.get("color_transfer"),
            "color_space": v.get("color_space"),
            "rotation": vtags.get("rotate"),
            "timecode": vtags.get("timecode") or out["timecode"],
        }
        if r_fps and a_fps and abs(r_fps - a_fps) > 0.05:
            out["warnings"].append(
                f"Variable frame rate likely (r={r_fps:.3f}, avg={a_fps:.3f}). Phone/screen footage? "
                "Run conform_footage to constant frame rate before editing or lip-sync will drift.")
        trc = (v.get("color_transfer") or "").lower()
        if trc in ("arib-std-b67",):
            out["warnings"].append("HLG HDR footage. Grade/tone-map before SDR delivery.")
        if trc in ("smpte2084",):
            out["warnings"].append("PQ HDR footage. Grade/tone-map before SDR delivery.")
        out["video"]["looks_like_log"] = _log_guess(v, tags, vtags)
    if a:
        out["audio"] = {
            "codec": a.get("codec_name"), "sample_rate": int(a.get("sample_rate", 0)),
            "channels": a.get("channels"),
        }
        if int(a.get("sample_rate", 0)) not in (48000,):
            out["warnings"].append(f"Audio is {a.get('sample_rate')} Hz; video deliverables expect 48 kHz.")
    else:
        out["warnings"].append("No audio stream: playback sync by waveform is impossible for this clip.")
    return out


def _log_guess(v: dict, tags: dict, vtags: dict) -> str:
    blob = " ".join(str(x) for x in [*tags.values(), *vtags.values()]).lower()
    for k in ("slog", "s-log", "vlog", "v-log", "clog", "c-log", "flog", "f-log", "nlog", "n-log", "logc", "log3g10", "dlog"):
        if k in blob:
            return f"yes (metadata mentions {k})"
    if (v.get("pix_fmt") or "").endswith("10le") and (v.get("color_transfer") in (None, "unknown", "bt709")):
        return "possibly (10-bit, no HDR transfer tag) - check the picture: flat/washed-out means LOG"
    return "unknown"


def decode_audio(path: str | Path, sr: int = 22050, start: float | None = None,
                 dur: float | None = None, mono: bool = True) -> np.ndarray:
    args = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    if dur:
        args += ["-t", f"{dur:.3f}"]
    args += ["-i", str(path), "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0:
        raise MediaError(p.stderr.decode(errors="ignore")[-800:])
    y = np.frombuffer(p.stdout, dtype=np.float32)
    if y.size == 0:
        raise MediaError(f"No audio could be decoded from {path}")
    return y


def loudness(path: str | Path, start: float | None = None, dur: float | None = None) -> dict:
    """Integrated LUFS, LRA, true peak via ebur128."""
    args = [FFMPEG, "-hide_banner", "-nostdin"]
    if start:
        args += ["-ss", str(start)]
    if dur:
        args += ["-t", str(dur)]
    args += ["-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"]
    p = subprocess.run(args, capture_output=True, text=True)
    s = p.stderr
    summary = s[s.rfind("Summary:"):] if "Summary:" in s else s

    def grab(pat):
        m = re.search(pat, summary)
        return float(m.group(1)) if m else None
    return {
        "integrated_lufs": grab(r"I:\s+(-?[\d.]+) LUFS"),
        "loudness_range_lu": grab(r"LRA:\s+(-?[\d.]+) LU"),
        "true_peak_dbtp": grab(r"Peak:\s+(-?[\d.]+) dBFS"),
    }


def seconds_to_tc(t: float, fps: float) -> str:
    """Non-drop timecode HH:MM:SS:FF using the nominal (rounded) frame rate."""
    nominal = int(round(fps))
    frames = int(round(t * fps))
    ff = frames % nominal
    s = frames // nominal
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{ff:02d}"


def tc_to_seconds(tc: str, fps: float) -> float:
    parts = re.split(r"[:;.]", tc)
    if len(parts) != 4:
        raise MediaError(f"Bad timecode {tc!r}; expected HH:MM:SS:FF")
    h, m, s, f = (int(x) for x in parts)
    nominal = int(round(fps))
    return ((h * 3600 + m * 60 + s) * nominal + f) / fps


def parse_time(t, fps: float = 24.0) -> float:
    """Accept seconds (number/str), 'MM:SS(.ms)', or 'HH:MM:SS:FF' timecode."""
    if isinstance(t, (int, float)):
        return float(t)
    t = str(t).strip()
    if t.count(":") == 3 or ";" in t:
        return tc_to_seconds(t, fps)
    if ":" in t:
        secs = 0.0
        for part in t.split(":"):
            secs = secs * 60 + float(part)
        return secs
    return float(t)
