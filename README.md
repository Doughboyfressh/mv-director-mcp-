# Music Video Director MCP

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Doughboyfressh/mv-director-mcp-?quickstart=1)

An MCP server that makes Claude your **musical director and finishing editor** for music videos.
It's built for videographers: you shot to playback, you've got a drive full of takes, and you need a
cut that sits on the beat, picture that's in sync, and a delivery package that passes platform specs.

Claude listens to the song, syncs every take by waveform (no slate needed), plans cuts on the
beat and bar grid, exports the result to your NLE, then finishes and QCs the deliverables.

## What it does

| Stage | Tools | What you get |
|---|---|---|
| **Ingest** | `inspect_footage` | Codec, fps, bit depth, colour tags, timecode. Flags **VFR phone footage**, **LOG/HDR**, 44.1k audio, and clips with no scratch audio. |
| **Listen** | `analyze_song` | BPM, beat grid, downbeats, **sections (intro / verse / pre / chorus / bridge / outro)**, key, energy per bar, the **impact moment** (biggest lift). Beat and bar lengths in *frames* at your timeline fps. |
| **Sync** | `sync_takes`, `create_project`, `set_clip` | Waveform sync of each take to the master, accurate to under a frame. Handles **overcranked takes shot to sped-up playback** (e.g. 2x playback at 48 fps). Gives a confidence score, alternative offsets, and coverage for each section. |
| **Conform** | `conform_footage` | VFR to CFR, slow-mo conform, ProRes Proxy/LT/422/HQ or DNxHR. Updates the project automatically. |
| **Direct** | `plan_edit`, `grab_frames` | Cut plan on the beat grid in 5 styles (performance, hype, cinematic, ballad, narrative) with a pace control. Picks takes by rotation, avoids same-size jump cuts, lands the hook wide, puts the hero shot on the impact moment, and adds director notes per section. `grab_frames` builds a storyboard contact sheet that Claude actually *looks at*. |
| **Hand-off** | `export_to_nle`, `render_rough_cut` | **FCPXML** (Resolve / Premiere / FCP) with the synced cut and section/bar markers, **CMX3600 EDL**, **Resolve marker EDL**, CSV. Also renders a labelled rough-cut MP4 for the artist. |
| **Finish** | `marry_master_audio`, `master_loudness`, `apply_look`, `add_lyrics` | Master track under picture with no video re-encode. Two-pass loudness for platform targets. LOG→709 `.cube` LUTs plus 8 looks, grain, vignette and letterbox, with a **before/after preview still**. Lyrics from LRC/SRT, burnt in and/or as a sidecar `.srt`, optionally snapped to the beat. |
| **Effects** | `detect_cuts`, `add_beat_effects` | Grades an edit against the music: % of cuts on the beat and on downbeats, plus a list of off-beat cuts with timecodes. Adds **Flash, Optical Shake, Zoom punch-out, Smooth Rotation, Smooth Slide (whip), Skipping Frames (stutter) and Night Vision**, placed on beats, downbeats, chorus hits, the drop, build-ups or your own cuts. Three ready styles (hype, smooth, subtle) or custom placements. Includes a dry run and a 12-second preview. Picture stays frame-exact and audio untouched. |
| **QC** | `qc_check` | Black/frozen frames mid-programme, silence drop-outs, clipping, loudness and true peak against spec, missing BT.709 tags, odd sizes, stream-length mismatch, and **frame-accurate sync check against the master**. |
| **Deliver** | `render_deliverables`, `make_social_cutdowns`, `make_spotify_canvas`, `pick_thumbnails` | ProRes HQ/4444 masters, YouTube 4K/1080, 9:16, 4:5, 1:1, watermarked client review, HEVC archive. Cutdowns that start and end on bar lines from the strongest hook. A Spotify Canvas loop that is a whole number of bars. Scored thumbnail candidates. |

Effects examples you can just say to Claude:
- "Add hype effects to my video" (style `hype`)
- "Night vision on the bridge, flashes on every chorus downbeat, shake on the drop"
- "Slide and zoom transitions only on cuts that land on the beat, every 2 bars"
- "Stutter the bar before each chorus"

Prompts: `finish_music_video` (full session kickoff) and `delivery_checklist` (QC + deliver a locked cut).

## Install

Requirements: **Python 3.10+** and **ffmpeg** (with ffprobe, libx264, prores_ks and libass, which standard builds include).

```bash
# macOS
brew install ffmpeg
# Windows
winget install ffmpeg
# Linux
sudo apt install ffmpeg
```

```bash
cd mv-director-mcp
pip install -e .            # or: uv pip install -e .
mv-director                 # starts the server on stdio (Ctrl-C to quit)
```

### Claude Desktop
Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):
```json
{
  "mcpServers": {
    "music-video-director": {
      "command": "uvx",
      "args": ["--from", "/ABSOLUTE/PATH/TO/mv-director-mcp", "mv-director"]
    }
  }
}
```
Without `uv`, use `"command": "python3", "args": ["-m", "mv_director"]` and add `"cwd"` pointing at the folder, or point `command` at the installed `mv-director` script.

### Claude Code
```bash
claude mcp add music-video-director -- uvx --from /ABSOLUTE/PATH/TO/mv-director-mcp mv-director
```

If ffmpeg isn't on PATH, set `MVD_FFMPEG` / `MVD_FFPROBE` environment variables in the config.

## Use it from your phone (free, with GitHub Codespaces)

The server can run in a free GitHub Codespace and connect to Claude as a custom connector. The Claude
phone app can then use it, and it downloads footage itself from Google Drive / Dropbox / OneDrive links.

**One-time setup (all from a phone browser):**
1. Open this repo on github.com → **Code** → **Codespaces** → **Create codespace on main**.
   The first build takes about 3–5 minutes (it installs ffmpeg).
2. When it finishes, the terminal shows **Your Claude connector URL**. It is also in `CONNECTOR_URL.txt`.
   If it says *ACTION NEEDED*, open the **Ports** tab and set port 8000 to **Public**.
3. In a browser, go to claude.ai → **Settings → Connectors → Add custom connector**. Name it
   "Music Video Director", paste the URL and save. It now shows up in the Claude phone app too.

**Each time you want to use it:** open github.com/codespaces and tap your codespace to start it.
The server starts automatically and the URL stays the same. Then chat with Claude on your phone:

> "Fetch my video from this Drive link, QC it for YouTube, and make 15 s vertical cutdowns: <link>"

Claude downloads the file onto the server, runs long jobs in the background, and gives you
**download links** for the results.

**Free-tier facts:**
- GitHub Free includes 120 core-hours a month. That is about 60 hours on the 2-core machine, plus 15 GB of storage.
- Codespaces stop after 30 minutes of inactivity. You can raise this to 4 hours in GitHub → Settings → Codespaces.
- A 2-core machine is fine for QC, analysis, cutdowns and 1080p renders. 4K ProRes renders are slow.
- Delete big files (or the whole codespace) when you're done so you stay under 15 GB.
- Drive files must be shared as **"Anyone with the link"** so the server can download them.

**Security:** the connector URL contains a random secret key, so treat it like a password.
Everyone without the key gets a 404. The server can only read and write inside its own workspace folder.

Other hosts: run `MVD_ACCESS_KEY=<long secret> MVD_PUBLIC_URL=https://your.host mv-director --http --port 8000`
on any machine with ffmpeg and HTTPS in front.

## A typical session

> "New project in ~/Jobs/Nova_Lights. Song is ~/Jobs/Nova_Lights/audio/NovaLights_master.wav, footage in
> ~/Jobs/Nova_Lights/A_CAM and B_CAM. 23.976 timeline, UHD. The B-cam slow-mo was shot at 59.94 with the track at 2.5x."

Claude will:
1. `create_project` with `playback_speeds={"B_CAM": 2.5}`. It reports the tempo, sections, and sync for every take, and flags problems ("A014 is VFR", "bridge has no performance coverage").
2. Fix what it finds: `conform_footage` for the VFR and overcranked clips, and `set_clip` for any low-confidence sync you check by eye.
3. `plan_edit` in the style you choose, then `grab_frames(every_cut=True)` to review the storyboard and give notes.
4. `export_to_nle` so you can open the FCPXML in Resolve or Premiere and finish the cut properly.
5. After you lock picture: `marry_master_audio`, `apply_look` (preview first), `add_lyrics`, `qc_check`, `render_deliverables`, `make_social_cutdowns`, `make_spotify_canvas`, `pick_thumbnails`.

## Tips for best results

- **Name your files with shot sizes** (`A001_WS_gimbal.mov`, `A007_CU_handheld.mov`). The planner reads WS/MS/CU/ECU/drone and gimbal/handheld/dolly/slowmo from filenames. It alternates sizes and avoids jump cuts.
- B-roll is detected from filename words such as `broll`, `story`, `insert`, `cutaway` or `texture`. Change these with `broll_keywords`, or per clip with `set_clip(role="broll")`.
- Sync needs scratch audio of the playback. Very repetitive songs can sync one or two bars off. The result lists **alternative offsets**: check one lip-sync line and use `set_clip(manual_offset=...)` if needed.
- Section names are heuristic (based on repetition and energy). Claude will tell you to confirm them by ear.
- Loudness: by default `master_loudness` only measures. Don't re-master a label's mix without approval; ask for a video master at about -14 LUFS instead.
- FCPXML import: Resolve → *File › Import › Timeline*. Premiere → *File › Import*. The song sits on an audio lane and camera audio on the synced clips is muted. Resolve marker EDL: right-click the timeline → *Timelines › Import › Timeline Markers from EDL* (timeline start 01:00:00:00).
- Long renders (4K ProRes, full deliverable sets) can take several minutes. If your client times out, render fewer presets per call.

## Limits (honest list)

- No RAW decode (BRAW/R3D/CRM). Transcode in your NLE or the camera vendor's tool first, then point the server at the ProRes.
- Overcranked takes are left out of FCPXML/EDL video tracks until they are conformed, which keeps NLE timing exact.
- Beat tracking assumes a steady tempo. Rubato or live-tempo songs get a usable but rougher grid.
- Tested with MCP Python SDK 2.x. A 1.x fallback import is included.

## Tests

```bash
python tests/make_fixtures.py /tmp/mvfx   # synthetic 124 BPM song + 4 fake camera takes (one 2x slow-mo)
python tests/test_e2e.py /tmp/mvfx        # runs every tool end-to-end (~4 min)
python tests/test_protocol.py             # talks to the server over real MCP stdio
MVD_ACCESS_KEY=... mv-director --http &    # then: python tests/test_remote.py <connector_url> <base url hosting test files>
```
