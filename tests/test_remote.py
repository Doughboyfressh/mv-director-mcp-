"""Remote/phone mode test: connect over HTTP with the secret URL, fetch media from links, run a background
job, get a download link. Usage: python tests/test_remote.py <connector_url> <link_base>"""
import asyncio, json, sys, urllib.request
from mcp import ClientSession
try:
    from mcp.client.streamable_http import streamable_http_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as http_client

URL, LINKS = sys.argv[1], sys.argv[2]


def data(res):
    d = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if d is None:
        d = json.loads(res.content[0].text)
    return d["result"] if isinstance(d, dict) and set(d) == {"result"} else d


async def main():
    async with http_client(URL) as streams:
        r, w = streams[0], streams[1]
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = [t.name for t in (await s.list_tools()).tools]
            print(len(names), "tools; remote tools present:", all(n in names for n in ("fetch_media", "start_job", "job_status", "list_workspace")))
            print("fetch song:", data(await s.call_tool("fetch_media", {"url": f"{LINKS}/song.wav"}))["relative_path"])
            j = data(await s.call_tool("start_job", {"tool": "fetch_media", "arguments": {"url": f"{LINKS}/A002_close.mp4"}}))
            for _ in range(60):
                st = data(await s.call_tool("job_status", {"job_id": j["job_id"]}))
                if st["status"] != "running":
                    break
                await asyncio.sleep(0.5)
            print("background fetch:", st["status"], st.get("result", {}).get("relative_path"))
            j = data(await s.call_tool("start_job", {"tool": "qc_check", "arguments": {
                "input_path": "inbox/A002_close.mp4", "master_song": "inbox/song.wav", "song_offset": "19.988"}}))
            for _ in range(240):
                st = data(await s.call_tool("job_status", {"job_id": j["job_id"]}))
                if st["status"] != "running":
                    break
                await asyncio.sleep(0.5)
            print("background QC:", st["status"], st.get("result", {}).get("verdict"), st.get("error"))
            res = await s.call_tool("analyze_song", {"song_path": "/etc/passwd"})
            print("sandbox blocks outside paths:", res.isError if hasattr(res, "isError") else getattr(res, "is_error", None))
            ws = data(await s.call_tool("list_workspace", {}))
            link = [f for f in ws["files"] if f["path"].endswith("song.wav")][0]["download_url"]
            n = len(urllib.request.urlopen(link).read())
            print("download link works:", n, "bytes")
asyncio.run(main())
