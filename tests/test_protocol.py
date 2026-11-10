"""Spawn the server over stdio and call tools through the real MCP protocol."""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "mv_director.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print("server:", (getattr(init, "server_info", None) or init.serverInfo).name, "| instructions:", bool(init.instructions))
            tools = (await s.list_tools()).tools
            print(len(tools), "tools:", ", ".join(t.name for t in tools))
            prompts = (await s.list_prompts()).prompts
            print("prompts:", [p.name for p in prompts])
            res = await s.call_tool("analyze_song", {"song_path": "/tmp/mvfx/song.wav", "fps": 23.976})
            d = (getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)) or json.loads(res.content[0].text)
            d = d["result"] if set(d) == {"result"} else d
            print("analyze_song ->", d["tempo_bpm"], "BPM,", d["key"]["key"], ", bar =", d["bar"]["frames"], "frames,",
                  [x["name"] for x in d["sections"]])
            res = await s.call_tool("sync_takes", {"song_path": "/tmp/mvfx/song.wav", "clips": ["/tmp/mvfx/A002_close.mp4"]})
            d = (getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)) or json.loads(res.content[0].text)
            d = d["result"] if set(d) == {"result"} else d
            print("sync_takes ->", d["results"][0]["offset_s"], d["results"][0]["song_in_tc"])
asyncio.run(main())
