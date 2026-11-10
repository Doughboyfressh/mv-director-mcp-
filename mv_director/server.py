"""Music Video Director - an MCP server for videographers finishing music videos."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import analysis as A
from . import director as D
from . import effects as FX
from . import nle as N
from . import remote as RM
from . import render as R
from .media import (MediaError, check_binaries, duration, ffmpeg, out_path, parse_time, probe, resolve,
                    seconds_to_tc, summarize)

VIDEO_EXT = {".mov", ".mp4", ".mxf", ".m4v", ".avi", ".mkv", ".mts", ".m2ts", ".braw", ".r3d", ".crm", ".insv"}
PROJECT_FILE = "mvd_project.json"

INSTRUCTIONS = """You are the Musical Director and finishing editor on a music video. The user is a videographer
(shooter/editor) who shot to playback and now needs to finish and deliver. Think like a music-video editor:
every cut lives on the musical grid (beats, bars, phrases), lip-sync must be frame-accurate, the hook gets the
best shots, and deliverables must pass platform specs.

Typical flow:
 1. create_project(song, footage folder, fps) -> analyses the song (tempo, downbeats, sections, key, impact
    moment), inspects every clip (VFR, LOG, HDR, 48k warnings) and syncs each take to the master by waveform.
    Ask the user for timeline fps and any overcranked/slow-mo playback speeds (e.g. 2x playback shot at 48fps).
 2. Review project_status: low-confidence syncs (check the alternatives), coverage gaps per section.
    conform_footage for VFR phone footage or overcranked takes; set_clip to fix roles/offsets.
 3. plan_edit(style, pace) -> beat-snapped cut plan with take selection + director notes per section.
 4. export_to_nle -> FCPXML (Resolve/Premiere/FCP), CMX3600 EDL, Resolve marker EDL, CSV. Editors finish in
    their NLE; render_rough_cut gives a quick assembly for the artist.
 5. Finishing: marry_master_audio (picture untouched), apply_look (preview a still first!), add_lyrics,
    master_loudness (measure first; don't re-master a label mix without approval).
 6. Effects: detect_cuts grades how tight an edit is to the beat; add_beat_effects adds Flash, Optical Shake,
    Zoom punch-outs, Smooth Rotation, Smooth Slide/whip, Skipping Frames stutter and Night Vision, timed to the
    music (dry_run first to show the timeline; preview a 12 s section before the full render).
 7. qc_check, then render_deliverables, make_social_cutdowns, make_spotify_canvas, pick_thumbnails.
REMOTE/PHONE MODE: files live in the server workspace. Get footage in with fetch_media(url) (Google Drive /
Dropbox / OneDrive share links or direct URLs), use relative paths like inbox/clip.mp4, and give the user
download_url links (list_workspace / get_download_link) for results. Run anything slow (fetch_media on big files,
renders, QC of a full video) with start_job and poll job_status - phone clients time out on long calls.
Use grab_frames to actually LOOK at footage/edits (returns image paths you can view) before giving creative notes.
Speak in editor language: timecode, frames, bars, ASL (average shot length). Offer opinions on pacing, shot
choice and story the way a director would, but the videographer has final cut. Long renders can take minutes.
"""

mcp = _Server("music-video-director", instructions=INSTRUCTIONS)


# ------------------------------------------------------------------ project helpers
def _pfile(project: str) -> Path:
    p = resolve(project, must_exist=False)
    return p if p.suffix == ".json" else p / PROJECT_FILE


def _load(project: str) -> dict:
    f = _pfile(project)
    if not f.exists():
        raise MediaError(f"No project at {f}. Run create_project first.")
    return json.loads(f.read_text())


def _save(project: str, data: dict) -> None:
    f = _pfile(project)
    f.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    f.write_text(json.dumps(data, indent=1))


def _collect(paths: list[str] | str) -> list[Path]:
    if isinstance(paths, str):
        paths = [paths]
    out: list[Path] = []
    for p in paths:
        rp = resolve(p)
        if rp.is_dir():
            out += sorted(x for x in rp.rglob("*") if x.suffix.lower() in VIDEO_EXT and not x.name.startswith("."))
        else:
            out.append(rp)
    return out


def _compact_analysis(a: dict, fps: float) -> dict:
    return {
        "tempo_bpm": a["tempo_bpm"], "tempo_alternatives": a["tempo_alternatives"], "key": a["key"],
        "duration": a["duration"], "duration_tc": seconds_to_tc(a["duration"], fps),
        "beats_per_bar": a["beats_per_bar"],
        "beat": {"seconds": a["beat_interval_s"], "frames": round(a["beat_interval_s"] * fps, 2)},
        "bar": {"seconds": a["bar_length_s"], "frames": round(a["bar_length_s"] * fps, 2)},
        "first_downbeat": a["first_downbeat"], "total_bars": len(a["bar_starts"]),
        "sections": [{**s, "start_tc": seconds_to_tc(s["start"], fps)} for s in a["sections"]],
        "impact_moment": a["impact_moment"], "dynamic_range_db": a["dynamic_range_db"], "notes": a["notes"],
    }



# ------------------------------------------------------------------ setup / inspection
@mcp.tool()
def check_setup() -> dict:
    """Verify ffmpeg/ffprobe are installed and list the presets this server offers."""
    return {"binaries": check_binaries(), "delivery_presets": {k: v["desc"] for k, v in R.DELIVERY_PRESETS.items()},
            "look_presets": list(R.LOOK_PRESETS), "loudness_targets": R.LOUDNESS_TARGETS,
            "edit_styles": {k: v["desc"] for k, v in D.STYLES.items()}, "conform_codecs": list(R.CONFORM_CODECS)}


@mcp.tool()
def inspect_footage(paths: list[str]) -> dict:
    """Inspect camera files or folders: codec, resolution, fps, bit depth, colour tags, timecode, LOG/HDR hints,
    and red flags (variable frame rate, 44.1k audio, no audio for sync)."""
    files = _collect(paths)
    res = []
    for f in files[:200]:
        try:
            res.append(summarize(f))
        except MediaError as e:
            res.append({"file": str(f), "error": str(e)})
    fps_set = sorted({r["video"]["fps"] for r in res if r.get("video")})
    return {"count": len(res), "frame_rates_found": fps_set,
            "mixed_frame_rates": len(fps_set) > 1, "files": res}


@mcp.tool()
def analyze_song(song_path: str, fps: float = 24.0, bpm_hint: float | None = None, beats_per_bar: int = 4,
                 save_json: str | None = None) -> dict:
    """Musical analysis of the master track: tempo, beat/bar grid, downbeats, sections (intro/verse/chorus...),
    key, energy, biggest 'impact' lift. Beat and bar lengths are also given in frames at `fps`.
    Full beat lists are saved to save_json (default: next to the song)."""
    a = A.analyze(str(resolve(song_path)), beats_per_bar=beats_per_bar, bpm_hint=bpm_hint)
    jp = out_path(save_json) if save_json else resolve(song_path).with_suffix(".analysis.json")
    jp.write_text(json.dumps(a, indent=1))
    return {**_compact_analysis(a, fps), "full_analysis_json": str(jp)}


@mcp.tool()
def sync_takes(song_path: str, clips: list[str], playback_speeds: dict[str, float] | None = None,
               fps: float = 24.0) -> dict:
    """Sync camera takes shot to playback against the master track by waveform (no slate needed).
    playback_speeds maps a filename (or substring) to the playback speed used on set, e.g. {"B001": 2.0}
    for a take shot at 48fps while the track played at 2x."""
    song = str(resolve(song_path))
    sdur = duration(probe(song))
    out = []
    for f in _collect(clips):
        spd = _speed_for(f, playback_speeds)
        r = A.sync_offset(song, str(f), playback_speed=spd)
        cd = duration(probe(f))
        r.update({"clip": f.name, "clip_duration": round(cd, 2)})
        if r.get("ok"):
            r["song_in_tc"] = seconds_to_tc(max(0, r["offset_s"]), fps)
            r["covers_song"] = [round(max(0, r["offset_s"]), 2), round(min(sdur, r["offset_s"] + cd * spd), 2)]
        out.append(r)
    return {"results": out, "tip": "Low confidence on very repetitive songs: check the alternatives "
                                   "(often exactly 1-2 bars apart) by eye on a vocal line, then set_clip(manual_offset)."}


def _speed_for(f: Path, speeds: dict[str, float] | None) -> float:
    for k, v in (speeds or {}).items():
        if k.lower() in f.name.lower():
            return float(v)
    return 1.0


# ------------------------------------------------------------------ project
@mcp.tool()
def create_project(project_dir: str, song_path: str, footage: list[str], fps: float = 23.976,
                   width: int = 3840, height: int = 2160, title: str = "Music Video", artist: str = "",
                   playback_speeds: dict[str, float] | None = None, broll_keywords: list[str] | None = None,
                   bpm_hint: float | None = None) -> dict:
    """One-shot setup: analyse the song, inspect + waveform-sync every take, and report coverage per section.
    broll_keywords: filename fragments that mark B-roll / story footage (default: broll, br_, story, insert, cutaway)."""
    song = resolve(song_path)
    a = A.analyze(str(song), bpm_hint=bpm_hint)
    smeta = summarize(song)
    kws = [k.lower() for k in (broll_keywords or ["broll", "b-roll", "br_", "story", "insert", "cutaway", "texture"])]
    clips = []
    for f in _collect(footage):
        try:
            meta = summarize(f)
        except MediaError as e:
            clips.append({"path": str(f), "error": str(e)})
            continue
        spd = _speed_for(f, playback_speeds)
        role = "broll" if any(k in f.name.lower() for k in kws) else "performance"
        sync = A.sync_offset(str(song), str(f), playback_speed=spd) if meta.get("audio") else {"ok": False, "reason": "no audio"}
        c = {"path": str(f), "media": meta, "role": role, "playback_speed": spd, "sync": sync}
        if sync.get("ok"):
            c["song_in"] = round(sync["offset_s"], 3)
            c["song_out"] = round(sync["offset_s"] + meta["duration_s"] * spd, 3)
        clips.append(c)
    proj = {"title": title, "artist": artist, "fps": fps, "width": width, "height": height,
            "song": str(song), "song_media": smeta, "analysis": a, "clips": clips, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(project_dir, proj)
    return {"project": str(_pfile(project_dir)), **project_status(project_dir)}


@mcp.tool()
def project_status(project: str) -> dict:
    """Song summary, every take's sync result and song coverage, per-section coverage, and problems to fix."""
    p = _load(project)
    fps = p["fps"]
    takes, problems = [], []
    for c in p["clips"]:
        name = Path(c["path"]).name
        if c.get("error"):
            problems.append(f"{name}: {c['error']}")
            continue
        s = c["sync"]
        row = {"clip": name, "role": c["role"], "playback_speed": c["playback_speed"],
               "fps": (c["media"].get("video") or {}).get("fps"), "excluded": c.get("excluded", False)}
        if s.get("ok"):
            row.update({"song_in_tc": seconds_to_tc(max(0, c["song_in"]), fps), "song_out_tc": seconds_to_tc(c["song_out"], fps),
                        "offset_s": s["offset_s"], "confidence": s["confidence"], "alternatives_s": s.get("alternatives_s")})
            if s["confidence"] == "low" and not s.get("manual"):
                problems.append(f"{name}: low sync confidence - verify lip-sync (alternatives {s.get('alternatives_s')})")
        else:
            row["sync"] = s.get("reason", "failed")
            if c["role"] == "performance":
                problems.append(f"{name}: not synced ({s.get('reason')}). Mark as broll or set manual_offset.")
        for w in c["media"].get("warnings", []):
            if "Variable frame rate" in w:
                problems.append(f"{name}: VFR - run conform_footage")
        vfps = (c["media"].get("video") or {}).get("fps")
        if vfps and abs(vfps - fps) > 0.05 and c["playback_speed"] == 1.0:
            problems.append(f"{name}: {vfps} fps in a {fps} fps timeline - if overcranked for slow-mo, set playback_speed; "
                            "otherwise it will be frame-rate converted")
        takes.append(row)
    cov = D.coverage(p["analysis"], [c for c in p["clips"] if not c.get("excluded") and not c.get("error")])
    for c in cov:
        if c["flag"]:
            problems.append(f"{c['section']} @ {seconds_to_tc(c['start'], fps)}: {c['flag']}")
    return {"title": p["title"], "fps": fps, "song": _compact_analysis(p["analysis"], fps), "takes": takes,
            "coverage": cov, "problems": problems, "has_cut_plan": "cut_plan" in p}


@mcp.tool()
def set_clip(project: str, clip: str, role: str | None = None, playback_speed: float | None = None,
             manual_offset: str | None = None, excluded: bool | None = None, resync: bool = False) -> dict:
    """Fix a take: role ('performance'|'broll'), playback_speed, manual_offset (song time at the clip's first
    frame, seconds or HH:MM:SS:FF), exclude it, or resync after changing speed."""
    p = _load(project)
    hit = [c for c in p["clips"] if clip.lower() in Path(c["path"]).name.lower()]
    if len(hit) != 1:
        raise MediaError(f"'{clip}' matched {len(hit)} clips")
    c = hit[0]
    if role:
        c["role"] = role
    if excluded is not None:
        c["excluded"] = excluded
    if playback_speed:
        c["playback_speed"] = playback_speed
    if resync:
        c["sync"] = A.sync_offset(p["song"], c["path"], playback_speed=c["playback_speed"])
    if manual_offset is not None:
        c["sync"] = {"ok": True, "offset_s": parse_time(manual_offset, p["fps"]), "confidence": "manual", "manual": True}
    if c["sync"].get("ok"):
        c["song_in"] = round(c["sync"]["offset_s"], 3)
        c["song_out"] = round(c["sync"]["offset_s"] + c["media"]["duration_s"] * c["playback_speed"], 3)
    _save(project, p)
    return {"clip": Path(c["path"]).name, "role": c["role"], "playback_speed": c["playback_speed"], "sync": c["sync"]}


@mcp.tool()
def conform_footage(input_path: str, output_path: str, fps: float = 23.976, playback_speed: float = 1.0,
                    codec: str = "prores_422", max_height: int | None = None, project: str | None = None) -> dict:
    """Make edit-ready media: variable->constant frame rate, overcranked slow-mo conform (playback_speed=2 for a
    take shot at 2x playback), and/or ProRes/DNxHR transcode. If project is given the take is swapped for
    the conformed file (sync offset preserved, speed reset to 1)."""
    r = R.conform(str(resolve(input_path)), str(out_path(output_path)), fps, playback_speed, codec, max_height)
    if project:
        p = _load(project)
        for c in p["clips"]:
            if Path(c["path"]) == resolve(input_path):
                c["original_path"] = c["path"]
                c["path"] = r["output"]
                c["media"] = summarize(r["output"])
                c["playback_speed"] = 1.0
                if c["sync"].get("ok"):
                    c["song_out"] = round(c["song_in"] + c["media"]["duration_s"], 3)
                r["project_updated"] = True
        _save(project, p)
    return r


# ------------------------------------------------------------------ editorial
@mcp.tool()
def plan_edit(project: str, style: str = "performance", pace: float = 1.0, min_shot_frames: int = 12,
              broll_ratio: float | None = None) -> dict:
    """Build a beat-snapped cut plan: where to cut (on beats/bars/section changes, hero cut on the impact
    moment), which synced take to use (rotating takes, avoiding same-size jump cuts, landing hooks wide),
    and director notes per section. style: performance|hype|cinematic|ballad|narrative. pace: 0.5 slower .. 2 faster."""
    p = _load(project)
    clips = [c for c in p["clips"] if not c.get("excluded") and not c.get("error")]
    plan = D.plan_cuts(p["analysis"], clips, p["fps"], style, pace, min_shot_frames, broll_ratio)
    p["cut_plan"] = plan
    _save(project, p)
    preview = plan["cuts"][:40]
    return {**{k: v for k, v in plan.items() if k != "cuts"},
            "cuts_preview": [{k: c[k] for k in ("index", "record_tc", "frames", "section", "beat_in_bar", "shot")} for c in preview],
            "note": f"{plan['total_cuts']} cuts saved to project. export_to_nle to take it into Resolve/Premiere/FCP."}


@mcp.tool()
def export_to_nle(project: str, output_dir: str | None = None,
                  formats: list[str] | None = None, include_cut: bool = True, include_bar_markers: bool = True,
                  lyrics: str | None = None) -> dict:
    """Write editor hand-off files: 'fcpxml' (FCP / DaVinci Resolve / Premiere import: song with section
    chapter markers + bar markers, and the synced cut on V1), 'edl' (CMX3600 cut list), 'resolve_markers'
    (Timeline > Import > Timeline Markers from EDL), 'csv'. lyrics: optional LRC/SRT path to add lyric markers."""
    p = _load(project)
    formats = formats or ["fcpxml", "edl", "resolve_markers", "csv"]
    od = out_path(output_dir) if output_dir else _pfile(project).parent / "nle_export"
    od.mkdir(parents=True, exist_ok=True)
    fps = p["fps"]
    plan = p.get("cut_plan") if include_cut else None
    cuts = plan["cuts"] if plan else []
    ly = R.parse_lyrics(lyrics) if lyrics else None
    markers = N.build_markers(p["analysis"], include_bars=include_bar_markers, lyrics=ly)
    safe = "".join(ch if ch.isalnum() else "_" for ch in p["title"])[:40]
    files, notes = {}, []
    clip_media = {c["path"]: c["media"] for c in p["clips"] if not c.get("error")}
    used = [c for c in p["clips"] if c["path"] in {x["clip"] for x in cuts if x["clip"]}]
    speed_clips = [c for c in used if c.get("playback_speed", 1) != 1]
    if speed_clips:
        notes.append("Overcranked takes (" + ", ".join(Path(c['path']).name for c in speed_clips) +
                     ") are left out of FCPXML/EDL video tracks - conform_footage them first so NLE timing is exact.")
    nle_cuts = [x for x in cuts if x["clip"] and x.get("playback_speed", 1) == 1]
    if "fcpxml" in formats:
        files["fcpxml"] = str(N.write_fcpxml(od / f"{safe}.fcpxml", p["title"], fps, p["width"], p["height"],
                                             p["song_media"], markers,
                                             clips=[{"path": c["path"], "media": c["media"]} for c in used if c.get("playback_speed", 1) == 1],
                                             cuts=nle_cuts))
    if "edl" in formats and nle_cuts:
        files["edl"] = str(N.write_cmx3600(od / f"{safe}_cut.edl", p["title"], fps, nle_cuts, clip_media, p["song_media"]))
    if "resolve_markers" in formats:
        files["resolve_markers"] = str(N.write_resolve_marker_edl(markers + N.build_markers(
            {"sections": [], "bar_starts": []}, include_bars=False, cut_plan=cuts), od / f"{safe}_markers.edl", fps, p["title"]))
    if "csv" in formats:
        files["csv"] = str(N.write_csv(markers, od / f"{safe}_markers.csv", fps))
        if cuts:
            import csv
            cp = od / f"{safe}_cutlist.csv"
            with open(cp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["#", "record_tc", "frames", "section", "beat_in_bar", "clip", "source_in_s", "shot"])
                for c in cuts:
                    w.writerow([c["index"], c["record_tc"], c["frames"], c["section"], c["beat_in_bar"],
                                Path(c["clip"]).name if c["clip"] else "", c["source_in"], c["shot"]])
            files["cutlist_csv"] = str(cp)
    notes.append("FCPXML: song is on an audio lane under a gap; camera audio is muted (-96 dB) on the synced clips. "
                 "Resolve: File > Import > Timeline. Premiere: File > Import the .fcpxml. Resolve markers EDL: "
                 "right-click timeline > Timelines > Import > Timeline Markers from EDL (timeline must start 01:00:00:00).")
    return {"files": files, "markers": len(markers), "cuts": len(nle_cuts), "notes": notes}


@mcp.tool()
def render_rough_cut(project: str, output_path: str, height: int = 1080, burn_labels: bool = True,
                     watermark: str | None = None) -> dict:
    """Render the current cut plan as a playable assembly with the master track married underneath.
    burn_labels overlays take name + record timecode for review sessions."""
    p = _load(project)
    plan = p.get("cut_plan")
    if not plan:
        raise MediaError("No cut plan - run plan_edit first")
    fps = p["fps"]
    W = int(round(p["width"] * height / p["height"])) // 2 * 2
    tmp = Path(tempfile.mkdtemp(prefix="mvd_rc_"))
    segs = []
    try:
        for c in plan["cuts"]:
            seg = tmp / f"{c['index']:05d}.mp4"
            length = c["end"] - c["start"]
            if length <= 0:
                continue
            vf_scale = f"scale={W}:{height}:force_original_aspect_ratio=decrease,pad={W}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            lab = ""
            if burn_labels:
                name = (Path(c["clip"]).stem if c["clip"] else "NO COVERAGE").replace("'", "").replace(":", " ")
                lab = (f",drawtext=text='{name}  |  {c['section']}':x=24:y=24:fontsize={height // 30}:fontcolor=white:box=1:boxcolor=black@0.5")
            if c["clip"]:
                spd = c.get("playback_speed") or 1.0
                src_in = max(0.0, c["source_in"])
                pad_head = max(0.0, -c["source_in"]) * spd
                vf = f"setpts=(PTS-STARTPTS)*{spd:.6f},fps={fps:.6f},{vf_scale}" + lab
                if pad_head:
                    vf += f",tpad=start_duration={pad_head:.3f}:start_mode=add:color=black"
                ffmpeg(["-ss", f"{src_in:.4f}", "-t", f"{length / spd + 0.5:.4f}", "-i", c["clip"], "-an", "-vf", vf,
                        "-t", f"{length:.4f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}", str(seg)])
            else:
                ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={W}x{height}:r={fps:.6f}:d={length:.4f}",
                        "-vf", "null" + lab, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(seg)])
            segs.append(seg)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in segs))
        picture = tmp / "picture.mp4"
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(picture)])
        outp = out_path(output_path)
        vf2 = []
        if burn_labels:
            vf2.append(f"drawtext=text='%{{pts\\:hms}}':x=w-tw-24:y=24:fontsize={height // 30}:fontcolor=white:box=1:boxcolor=black@0.5")
        if watermark:
            wm = watermark.replace("'", "").replace(":", " ")
            vf2.append(f"drawtext=text='{wm}':x=(w-tw)/2:y=h-th-40:fontsize={height // 20}:fontcolor=white@0.3")
        ffmpeg(["-i", str(picture), "-i", p["song"], "-map", "0:v", "-map", "1:a",
                *(["-vf", ",".join(vf2), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"] if vf2 else ["-c:v", "copy"]),
                *R.REC709, "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(outp)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"output": str(outp), "cuts": len(segs), "duration_s": round(duration(probe(outp)), 2)}


# ------------------------------------------------------------------ finishing
@mcp.tool()
def marry_master_audio(video_path: str, song_path: str, output_path: str, song_offset: str = "0",
                       fade_in: float = 0.0, fade_out: float = 0.0, length: str = "video") -> dict:
    """Replace the timeline/camera audio with the master track without re-encoding picture.
    song_offset: song time at picture frame 0 (seconds or timecode; negative if picture starts before the song).
    .mov output gets 24-bit PCM, .mp4 gets AAC 320k."""
    return R.marry_audio(str(resolve(video_path)), str(resolve(song_path)), str(out_path(output_path)),
                         parse_time(song_offset), fade_in, fade_out, length)


@mcp.tool()
def master_loudness(input_path: str, output_path: str | None = None, target: str = "youtube",
                    true_peak: float | None = None) -> dict:
    """Measure (no output_path) or normalise programme loudness for a platform: youtube, spotify, apple_music,
    instagram, tiktok, tidal, ebu_r128_broadcast, atsc_a85_broadcast, cinema_web - or a LUFS number as string.
    Uses two-pass linear loudnorm (dynamics preserved when possible); picture is stream-copied."""
    tgt: Any = float(target) if target.lstrip("-").replace(".", "").isdigit() else target
    return R.master_loudness(str(resolve(input_path)), str(out_path(output_path)) if output_path else None, tgt, true_peak)


@mcp.tool()
def apply_look(input_path: str, output_path: str, lut_path: str | None = None, lut_strength: float = 1.0,
               preset: str = "none", exposure: float = 0.0, contrast: float = 1.0, saturation: float = 1.0,
               temperature: float = 0.0, tint: float = 0.0, grain: float = 0.0, vignette: float = 0.0,
               letterbox: float | None = None, sharpen: float = 0.0, preview_at: str | None = None) -> dict:
    """Grade: camera/creative .cube LUT (e.g. S-Log3 -> Rec.709) at lut_strength, then exposure (stops),
    contrast/saturation (1.0 = unchanged), temperature/tint (-1..1), a look preset (warm_film, teal_orange,
    bleach_bypass, noir_bw, vintage_fade, night_cool, neon_pop, clean), grain/vignette (0..1),
    letterbox aspect (2.39, 2.0, 1.85). Set preview_at (seconds/TC) to get a BEFORE|AFTER still first - look at it."""
    kw = dict(lut=str(resolve(lut_path)) if lut_path else None, lut_strength=lut_strength, preset=preset,
              exposure=exposure, contrast=contrast, saturation=saturation, temperature=temperature, tint=tint,
              grain=grain, vignette=vignette, letterbox=letterbox, sharpen=sharpen)
    return R.apply_look(str(resolve(input_path)), str(out_path(output_path)),
                        preview_at=parse_time(preview_at) if preview_at is not None else None, **kw)


@mcp.tool()
def add_lyrics(video_path: str, lyrics: str, output_path: str, burn_in: bool = True, font: str = "Inter",
               size_pct: float = 5.0, position: str = "bottom", color: str = "FFFFFF", uppercase: bool = False,
               song_offset: str = "0", snap_to_beats_project: str | None = None) -> dict:
    """Lyrics as styled burnt-in captions and/or sidecar .srt (for YouTube CC). lyrics: path to .lrc/.srt or the
    LRC text itself ([mm:ss.xx] line). song_offset = song time at video frame 0. position: bottom|center|top|
    lower_left|lower_right. snap_to_beats_project: snap line starts to the nearest beat from that project."""
    vp = resolve(video_path)
    info = probe(vp)
    from .media import video_stream
    v = video_stream(info)
    lines = R.parse_lyrics(lyrics)
    if not lines:
        raise MediaError("No timed lyric lines found (need LRC [mm:ss.xx] or SRT)")
    if snap_to_beats_project:
        lines = R.snap_lyrics(lines, _load(snap_to_beats_project)["analysis"]["beats"])
    off = parse_time(song_offset)
    outp = out_path(output_path)
    srt = R.write_srt(lines, outp.with_suffix(".srt"), off)
    ass = R.write_ass(lines, outp.with_suffix(".ass"), v["width"], v["height"], font, size_pct, position, color,
                      2.5, uppercase, off)
    res = {"lines": len(lines), "srt": str(srt), "ass": str(ass)}
    if burn_in:
        R.burn_subtitles(str(vp), ass, str(outp))
        res["output"] = str(outp)
    return res


# ------------------------------------------------------------------ delivery
@mcp.tool()
def render_deliverables(input_path: str, output_dir: str, presets: list[str] | None = None,
                        focus_x: float = 0.5, focus_y: float = 0.5, title: str | None = None,
                        artist: str | None = None, watermark: str | None = None) -> dict:
    """Render the delivery package from the finished master. Presets: master_prores_hq, master_prores_4444,
    youtube_4k, youtube_1080, vertical_9x16, instagram_4x5, square_1x1, client_review, archive_hevc_10bit.
    focus_x/focus_y (0..1) choose where vertical/square crops sit (0.5 = centre, 0.3 = left third).
    All renders are tagged BT.709 with closed GOPs and faststart."""
    presets = presets or ["youtube_1080", "vertical_9x16", "client_review"]
    inp = str(resolve(input_path))
    od = out_path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    stem = Path(inp).stem
    meta = {k: v for k, v in (("title", title), ("artist", artist)) if v}
    out = []
    for pr in presets:
        ext = R.DELIVERY_PRESETS.get(pr, {}).get("ext", "mp4")
        try:
            out.append(R.render_preset(inp, str(od / f"{stem}_{pr}.{ext}"), pr, focus_x, focus_y,
                                       watermark=watermark if pr == "client_review" else None, metadata=meta))
        except MediaError as e:
            out.append({"preset": pr, "error": str(e)})
    return {"deliverables": out}


@mcp.tool()
def make_social_cutdowns(input_path: str, output_dir: str, project: str, lengths: list[int] | None = None,
                         presets: list[str] | None = None, section: str = "auto", song_offset: str = "0",
                         focus_x: float = 0.5) -> dict:
    """Teasers/cutdowns that start and end on bar lines: picks the strongest hook (or a named section),
    starts one bar early for a pickup, and renders each length x aspect with short fades.
    song_offset = song time at the video's frame 0 (0 for a finished video)."""
    p = _load(project)
    lengths = lengths or [15, 30]
    presets = presets or ["vertical_9x16"]
    off = parse_time(song_offset)
    od = out_path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    out = []
    for L in lengths:
        s, e, why = R.best_window(p["analysis"], L, section)
        for pr in presets:
            ext = R.DELIVERY_PRESETS[pr]["ext"]
            r = R.render_preset(str(resolve(input_path)), str(od / f"{Path(input_path).stem}_{L}s_{pr}.{ext}"), pr,
                                focus_x=focus_x, start=s - off, length=e - s, fade=0.25)
            r.update({"song_window": [round(s, 2), round(e, 2)], "why": why})
            out.append(r)
    return {"cutdowns": out}


@mcp.tool()
def make_spotify_canvas(input_path: str, output_path: str, project: str, start: str | None = None,
                        bars: int | None = None, song_offset: str = "0", focus_x: float = 0.5) -> dict:
    """Spotify Canvas loop: 9:16, 3-8 s, no audio, length = whole bars so it loops in time with the song.
    Starts on the impact moment / hook downbeat unless start is given."""
    p = _load(project)
    a = p["analysis"]
    bar = a["bar_length_s"]
    if not bars:
        bars = max(1, min(int(8 // bar), max(1, round(6 / bar))))
    length = bars * bar
    if length < 3:
        bars = int(-(-3 // bar))
        length = bars * bar
    if length > 8:
        length = 8.0
    if start is not None:
        s = parse_time(start)
    else:
        s = (a.get("impact_moment") or {}).get("time") or R.best_window(a, length)[0]
    r = R.render_preset(str(resolve(input_path)), str(out_path(output_path)), "vertical_9x16", focus_x=focus_x,
                        start=s - parse_time(song_offset), length=length, no_audio=True)
    r.update({"bars": bars, "loop_length_s": round(length, 3), "song_start": round(s, 3),
              "spec": "Canvas: 9:16, 3-8s, min 720px tall, no audio, no text/logos/rapid flashing."})
    return r


@mcp.tool()
def qc_check(input_path: str, spec: str = "youtube", master_song: str | None = None, song_offset: str = "0") -> dict:
    """Pre-delivery QC: codec/pixel format/colour tags/frame rate/VFR, black or frozen frames mid-programme,
    silence drop-outs, clipping, loudness + true peak vs spec (youtube|streaming_social|label_master|broadcast_ebu),
    stream length mismatch, and (with master_song) frame-accurate audio sync against the master."""
    return R.qc(str(resolve(input_path)), spec, str(resolve(master_song)) if master_song else None, parse_time(song_offset))


@mcp.tool()
def pick_thumbnails(input_path: str, output_dir: str, count: int = 6, project: str | None = None,
                    song_offset: str = "0") -> dict:
    """Find thumbnail candidates: samples the hook downbeats and impact moment (if a project is given) plus an
    even spread, scores sharpness/exposure/contrast, exports the best stills and a labelled contact sheet."""
    inp = str(resolve(input_path))
    dur = duration(probe(inp))
    off = parse_time(song_offset)
    times = list(range(2, int(dur) - 1, max(1, int(dur // 40))))
    if project:
        a = _load(project)["analysis"]
        for s in a["sections"]:
            if s.get("name") == "chorus":
                times += [b - off for b in a["bar_starts"] if s["start"] <= b < s["end"]]
        if a.get("impact_moment"):
            times.append(a["impact_moment"]["time"] - off)
    times = sorted({round(t + 0.12, 2) for t in times if 0 < t < dur - 0.2})
    scored = []
    for t in times:
        g = R._gray(inp, t)
        if g is not None:
            scored.append((R.score_frame(g), t))
    scored.sort(key=lambda x: -x[0]["score"])
    picks, last = [], []
    for sc, t in scored:
        if all(abs(t - u) > max(2.0, dur / (count * 3)) for u in last):
            picks.append((sc, t))
            last.append(t)
        if len(picks) >= count:
            break
    od = out_path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (sc, t) in enumerate(picks, 1):
        f = od / f"thumb_{i:02d}_{seconds_to_tc(t, 24).replace(':', '-')}.jpg"
        R.grab_frame(inp, t, str(f), width=1920)
        files.append({"file": str(f), "time": t, "sharpness": round(sc["sharpness"], 1), "brightness": round(sc["brightness"])})
    sheet = R.contact_sheet([x["file"] for x in files], str(od / "thumbnails_contact_sheet.jpg"), cols=3,
                            labels=[f"#{i + 1} {x['time']:.1f}s" for i, x in enumerate(files)])
    return {"thumbnails": files, "contact_sheet": sheet,
            "tip": "Open the contact sheet. Best music-video thumbnails: artist's face, eyes visible, strong colour, room for title text."}


@mcp.tool()
def grab_frames(input_path: str, output_path: str, times: list[str] | None = None, project: str | None = None,
                every_cut: bool = False, cols: int = 4) -> dict:
    """Look at footage: grab stills at given times (seconds or TC) - or, with project + every_cut, the first frame
    of every cut in the plan (a storyboard of the edit) - and tile them into one labelled contact sheet image.
    Read the returned image to give visual notes on framing, continuity and grade."""
    inp = str(resolve(input_path)) if input_path else None
    tmp = Path(tempfile.mkdtemp(prefix="mvd_grab_"))
    imgs, labels = [], []
    try:
        if every_cut and project:
            p = _load(project)
            for c in p.get("cut_plan", {}).get("cuts", [])[:60]:
                if not c["clip"]:
                    continue
                f = tmp / f"{c['index']:04d}.jpg"
                R.grab_frame(c["clip"], max(0, c["source_in"]) + 0.05, str(f), width=640)
                imgs.append(str(f))
                labels.append(f"{c['index']} {c['record_tc']} {Path(c['clip']).stem[:14]}")
        else:
            fps = 24.0
            for i, t in enumerate(times or []):
                ts = parse_time(t, fps)
                f = tmp / f"{i:04d}.jpg"
                R.grab_frame(inp, ts, str(f), width=960)
                imgs.append(str(f))
                labels.append(f"{ts:.2f}s")
        if not imgs:
            raise MediaError("Nothing to grab - give times, or project + every_cut")
        outp = out_path(output_path)
        R.contact_sheet(imgs, str(outp), cols=cols, labels=labels)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"image": str(outp), "frames": len(imgs)}


# ------------------------------------------------------------------ beat-synced effects
_ANALYSIS_CACHE: dict[str, dict] = {}


def _analysis_for(video: Path, project: str | None, song_path: str | None) -> dict:
    if project:
        return _load(project)["analysis"]
    src = str(resolve(song_path)) if song_path else str(video)
    key = f"{src}:{Path(src).stat().st_mtime}"
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = A.analyze(src)
    return _ANALYSIS_CACHE[key]


@mcp.tool()
def detect_cuts(video_path: str, project: str | None = None, song_path: str | None = None,
                song_offset: str = "0", sensitivity: float = 10.0) -> dict:
    """Find every hard cut in an edited video and grade the edit against the music: % of cuts on the beat
    (within 1-2 frames), % on downbeats, average shot length, and a list of off-beat cuts with timecodes.
    Uses the project's song analysis, a separate song file, or the video's own soundtrack."""
    vp = resolve(video_path)
    info = probe(vp)
    from .media import fps_of
    fps = fps_of(info) or 24.0
    a = _analysis_for(vp, project, song_path)
    off = parse_time(song_offset) if (project or song_path) else 0.0
    cuts = FX.detect_cuts(str(vp), sensitivity)
    rep = FX.cut_report(cuts, a, off, fps, duration(info))
    rep["song"] = {"tempo_bpm": a["tempo_bpm"], "sections": [(s.get("name"), s["start"]) for s in a["sections"]]}
    return rep


@mcp.tool()
def add_beat_effects(video_path: str, output_path: str, style: str | None = "hype",
                     effects: list[dict] | None = None, project: str | None = None, song_path: str | None = None,
                     song_offset: str = "0", preview_start: str | None = None, preview_seconds: float | None = None,
                     dry_run: bool = False) -> dict:
    """Add music-synced effects to an edited video: flash, shake (optical shake), zoom (punch-out
    transition), rotation (smooth spin transition), slide (whip / smooth slide), skip (skipping-frames
    stutter) and nightvision (look).
    style: hype | smooth | subtle (ready-made recipes), or None to use only `effects`.
    effects: extra placements, e.g. [{"type":"nightvision","where":"section:bridge"},
      {"type":"flash","where":"downbeats","intensity":0.4,"every":4}, {"type":"transitions","where":"cuts_on_beat"}].
    Point placements: impact, downbeats, beats, chorus_downbeats, chorus_beats, chorus_starts, section_starts,
      hi_energy_beats, cuts, cuts_on_beat, chorus_cuts, section:<name>, or a list of seconds (song time).
    Window placements (skip, nightvision): build_ups, chorus, impact, section:<name>, range:<a>-<b>, [[a,b],...].
    Beats come from the project, a song file, or the video's own soundtrack. dry_run=True returns the
    effect timeline without rendering; preview_start/preview_seconds render a short test section first."""
    vp = resolve(video_path)
    info = probe(vp)
    from .media import fps_of
    fps = fps_of(info) or 24.0
    vdur = duration(info)
    a = _analysis_for(vp, project, song_path)
    off = parse_time(song_offset) if (project or song_path) else 0.0
    cuts = FX.detect_cuts(str(vp))
    events = FX.plan(a, cuts, off, fps, vdur, style, effects)
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    res = {"style": style, "style_brief": FX.STYLE_DESC.get(style or "", "custom"), "tempo_bpm": a["tempo_bpm"],
           "cuts_detected": len(cuts), "effect_counts": counts, "events": events[:150]}
    if dry_run:
        return res
    start = parse_time(preview_start) if preview_start is not None else None
    length = preview_seconds or (None if start is None else 12.0)
    r = FX.render(str(vp), str(out_path(output_path)), events, start, length)
    res.update(r)
    res["download_url"] = RM.download_link(r["output"])
    return res


# ------------------------------------------------------------------ workspace / remote
@mcp.tool()
def fetch_media(url: str, subfolder: str = "inbox", filename: str | None = None) -> dict:
    """Download footage or a song onto the server from a share link: Google Drive ('Anyone with the link'),
    Dropbox, OneDrive, or any direct URL. For big files use start_job("fetch_media", {...}) and poll job_status."""
    r = RM.fetch(url, subfolder, filename)
    try:
        r["media"] = summarize(r["path"])
    except MediaError:
        pass
    r["download_url"] = RM.download_link(r["path"])
    return r


@mcp.tool()
def list_workspace(subfolder: str = "") -> dict:
    """List files in the server workspace with sizes and download links (remote mode)."""
    return {"workspace": str(RM.WORKDIR) if RM.REMOTE else os.getcwd(), "files": RM.list_files(subfolder)}


@mcp.tool()
def get_download_link(path: str) -> dict:
    """A link the user can tap on their phone to download a result file (remote mode only)."""
    p = resolve(path)
    link = RM.download_link(p)
    if not link:
        raise MediaError("Download links only exist when the server runs in remote mode with MVD_PUBLIC_URL set.")
    return {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1), "download_url": link}


@mcp.tool()
def start_job(tool: str, arguments: dict) -> dict:
    """Run any other tool in the background (e.g. tool="render_deliverables", arguments={...}) and return a
    job id immediately. Use for downloads, renders, QC and project creation on full-length videos."""
    if tool in ("start_job", "job_status"):
        raise MediaError("Can't background that tool")
    fn = globals().get(tool)
    if not callable(fn):
        raise MediaError(f"Unknown tool {tool}")
    return RM.start_job(tool, fn, dict(arguments or {}))


@mcp.tool()
def job_status(job_id: str | None = None) -> dict:
    """Progress/result of a background job (omit job_id to list recent jobs)."""
    return RM.job_status(job_id)


# ------------------------------------------------------------------ prompts
@mcp.prompt()
def finish_music_video(song_path: str, footage_folder: str, project_dir: str) -> str:
    """Start a finishing session as the musical director."""
    return (f"Act as musical director and finishing editor. Song: {song_path}. Footage: {footage_folder}. "
            f"Project folder: {project_dir}. First ask me the timeline frame rate, delivery resolution, any takes shot "
            "to sped-up playback, and the vibe/edit style. Then create_project, walk me through tempo/sections/"
            "coverage problems, propose an edit style and pace with reasons, plan_edit, look at the storyboard with "
            "grab_frames, and export to my NLE. After I lock picture, run finishing, QC and deliverables.")


@mcp.prompt()
def delivery_checklist(master_path: str, song_path: str) -> str:
    """QC and deliver a locked music video."""
    return (f"Locked master: {master_path}. Master song: {song_path}. Run qc_check (youtube spec, with master_song), "
            "explain every WARN/FAIL in plain terms with the fix, measure loudness, then render_deliverables "
            "(master_prores_hq, youtube_4k or youtube_1080, vertical_9x16, client_review), social cutdowns, "
            "a Spotify Canvas and thumbnail picks. Finish with a delivery sheet listing every file and its spec.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Music Video Director MCP server")
    ap.add_argument("--http", action="store_true", help="serve over HTTP for remote/phone use")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    if not args.http:
        mcp.run()
        return
    if len(RM.ACCESS_KEY) < 16:
        raise SystemExit("Set MVD_ACCESS_KEY to a long random secret (16+ chars) before serving over HTTP.")
    RM.enable_remote()
    import uvicorn
    app = RM.build_app(mcp, args.host)
    base = RM.PUBLIC_URL or f"http://{args.host}:{args.port}"
    print(f"\nMusic Video Director is running.\nConnector URL: {base}/{RM.ACCESS_KEY}/mcp\n"
          f"Workspace: {RM.WORKDIR}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
