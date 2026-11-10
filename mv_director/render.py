"""Rendering / finishing operations built on ffmpeg."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .media import (FFMPEG, MediaError, audio_stream, duration, ffmpeg, fps_of, loudness, probe,
                    seconds_to_tc, video_stream)

REC709 = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]

# ------------------------------------------------------------------ platform specs
LOUDNESS_TARGETS = {
    "youtube": (-14.0, -1.0), "spotify": (-14.0, -1.0), "apple_music": (-16.0, -1.0),
    "instagram": (-14.0, -1.0), "tiktok": (-14.0, -1.0), "tidal": (-14.0, -1.0),
    "ebu_r128_broadcast": (-23.0, -1.0), "atsc_a85_broadcast": (-24.0, -2.0), "cinema_web": (-16.0, -1.0),
}

DELIVERY_PRESETS = {
    "master_prores_hq": {"ext": "mov", "desc": "Archival/label master. ProRes 422 HQ 10-bit, 24-bit PCM.",
                         "v": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"],
                         "a": ["-c:a", "pcm_s24le", "-ar", "48000"], "aspect": None},
    "master_prores_4444": {"ext": "mov", "desc": "ProRes 4444 for VFX/graphics hand-off.",
                           "v": ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"],
                           "a": ["-c:a", "pcm_s24le", "-ar", "48000"], "aspect": None},
    "youtube_4k": {"ext": "mp4", "desc": "YouTube 2160p upload. H.264 High, CRF 16, closed GOP, 384k AAC.",
                   "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-pix_fmt", "yuv420p", "-bf", "2"],
                   "a": ["-c:a", "aac", "-b:a", "384k", "-ar", "48000"], "aspect": None, "height": 2160},
    "youtube_1080": {"ext": "mp4", "desc": "YouTube 1080p upload. H.264 High, CRF 17.",
                     "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-bf", "2"],
                     "a": ["-c:a", "aac", "-b:a", "384k", "-ar", "48000"], "aspect": None, "height": 1080},
    "vertical_9x16": {"ext": "mp4", "desc": "Reels / TikTok / Shorts 1080x1920.",
                      "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p"],
                      "a": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"], "aspect": (9, 16), "size": (1080, 1920)},
    "instagram_4x5": {"ext": "mp4", "desc": "Instagram feed 1080x1350.",
                      "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p"],
                      "a": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"], "aspect": (4, 5), "size": (1080, 1350)},
    "square_1x1": {"ext": "mp4", "desc": "Square 1080x1080 (feeds, ads).",
                   "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p"],
                   "a": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"], "aspect": (1, 1), "size": (1080, 1080)},
    "client_review": {"ext": "mp4", "desc": "Small 720p review copy with optional watermark + burnt-in timecode.",
                      "v": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"],
                      "a": ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"], "aspect": None, "height": 720, "review": True},
    "archive_hevc_10bit": {"ext": "mp4", "desc": "Space-saving 10-bit HEVC archive copy.",
                           "v": ["-c:v", "libx265", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"],
                           "a": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"], "aspect": None},
}

LOOK_PRESETS = {
    "none": "",
    "clean": "eq=contrast=1.05:saturation=1.06",
    "warm_film": "colorbalance=rs=0.04:gs=0.0:bs=-0.04:rm=0.03:bm=-0.03:rh=0.02:bh=-0.03,"
                 "curves=all='0/0.035 0.25/0.23 0.75/0.78 1/0.96',eq=saturation=0.92",
    "teal_orange": "colorbalance=rs=-0.08:gs=0.0:bs=0.08:rh=0.07:gh=0.01:bh=-0.07,eq=contrast=1.08:saturation=1.1",
    "bleach_bypass": "eq=contrast=1.28:saturation=0.45:gamma=0.97,curves=all='0/0 0.5/0.47 1/1'",
    "noir_bw": "hue=s=0,eq=contrast=1.25,curves=all='0/0 0.3/0.22 0.7/0.8 1/1'",
    "vintage_fade": "curves=all='0/0.08 0.5/0.5 1/0.9',colorbalance=rs=0.05:bs=-0.03:rh=0.04,eq=saturation=0.8",
    "night_cool": "colorbalance=rs=-0.05:bs=0.08:rm=-0.03:bm=0.05,eq=contrast=1.1:saturation=0.9:gamma=0.95",
    "neon_pop": "eq=contrast=1.12:saturation=1.35,colorbalance=rh=0.03:bh=0.05:bs=0.04",
}


def _vinfo(path) -> tuple[dict, dict | None, dict | None]:
    info = probe(path)
    return info, video_stream(info), audio_stream(info)


def _reframe(w: int, h: int, aspect: tuple[int, int] | None, size: tuple[int, int] | None,
             focus_x: float = 0.5, focus_y: float = 0.5, height: int | None = None) -> str:
    """Crop to aspect around a focus point, then scale. Never upscales unless size forces it."""
    f = []
    if aspect:
        ar = aspect[0] / aspect[1]
        if w / h > ar:   # too wide: crop width
            cw, ch = int(h * ar) // 2 * 2, h
        else:
            cw, ch = w, int(w / ar) // 2 * 2
        x = int((w - cw) * min(max(focus_x, 0), 1))
        y = int((h - ch) * min(max(focus_y, 0), 1))
        f.append(f"crop={cw}:{ch}:{x}:{y}")
    if size:
        f.append(f"scale={size[0]}:{size[1]}:flags=lanczos")
    elif height and h != height:
        f.append(f"scale=-2:{height}:flags=lanczos")
    f.append("setsar=1")
    return ",".join(f)


# ------------------------------------------------------------------ audio marriage & loudness
def marry_audio(video: str, song: str, output: str, song_offset: float = 0.0, fade_in: float = 0.0,
                fade_out: float = 0.0, length: str = "video", audio_codec: str | None = None) -> dict:
    """Lay the master track under the picture. song_offset = song time at picture frame 0."""
    info, v, _ = _vinfo(video)
    vdur = duration(info)
    sdur = duration(probe(song))
    ext = Path(output).suffix.lower()
    ac = audio_codec or ("pcm_s24le" if ext in (".mov", ".mxf") else "aac")
    a_args = ["-c:a", ac, "-ar", "48000"] + (["-b:a", "320k"] if ac == "aac" else [])
    af = []
    s_in = []
    if song_offset > 0:
        s_in = ["-ss", f"{song_offset:.4f}"]
    elif song_offset < 0:
        ms = int(round(-song_offset * 1000))
        af.append(f"adelay={ms}|{ms}")
    total = vdur if length == "video" else min(vdur, sdur - max(0, song_offset))
    if fade_in > 0:
        af.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        af.append(f"afade=t=out:st={max(0, total - fade_out):.3f}:d={fade_out}")
    args = ["-i", video, *s_in, "-i", song, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *a_args,
            "-t", f"{total:.4f}"]
    if af:
        args += ["-af", ",".join(af)]
    if ext in (".mp4", ".m4v", ".mov"):
        args += ["-movflags", "+faststart"]
    ffmpeg(args + [output])
    return {"output": output, "duration_s": round(total, 3), "audio_codec": ac, "song_offset": song_offset,
            "video": "stream-copied (no generation loss)"}


def master_loudness(inp: str, output: str | None, target: str | float = "youtube", true_peak: float | None = None,
                    lra: float = 11.0) -> dict:
    if isinstance(target, str):
        if target not in LOUDNESS_TARGETS:
            raise MediaError(f"Unknown target {target}. Options: {list(LOUDNESS_TARGETS)} or a LUFS number")
        I, TP = LOUDNESS_TARGETS[target]
    else:
        I, TP = float(target), -1.0
    TP = true_peak if true_peak is not None else TP
    before = loudness(inp)
    report = {"target_lufs": I, "target_true_peak": TP, "before": before}
    if not output:
        delta = None if before["integrated_lufs"] is None else round(I - before["integrated_lufs"], 1)
        report["gain_needed_db"] = delta
        report["advice"] = _loudness_advice(before, I, TP)
        return report
    p = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", inp, "-vn", "-af",
                        f"loudnorm=I={I}:TP={TP}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise MediaError("loudnorm measurement failed")
    j = json.loads(m.group(0))
    lin = "true" if float(j["input_lra"]) <= lra else "false"
    af = (f"loudnorm=I={I}:TP={TP}:LRA={lra}:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear={lin},"
          f"aresample=48000")
    info, v, _ = _vinfo(inp)
    ext = Path(output).suffix.lower()
    ac = ["-c:a", "pcm_s24le"] if ext in (".mov", ".wav", ".mxf") else ["-c:a", "aac", "-b:a", "320k"]
    vargs = ["-map", "0:v:0", "-c:v", "copy"] if v else []
    ffmpeg(["-i", inp, *vargs, "-map", "0:a:0", "-af", af, *ac, output])
    report["after"] = loudness(output)
    report["mode"] = "linear gain (song dynamics preserved)" if lin == "true" else "dynamic (song LRA exceeded target LRA)"
    report["output"] = output
    return report


def _loudness_advice(m: dict, I: float, TP: float) -> str:
    li, tp = m.get("integrated_lufs"), m.get("true_peak_dbtp")
    if li is None:
        return "Could not measure."
    out = []
    if li > I + 1:
        out.append(f"Louder than target by {li - I:.1f} LU - the platform will turn it down; no action strictly needed, "
                   "but if the label wants the video to match streaming, normalise.")
    elif li < I - 1:
        out.append(f"Quieter than target by {I - li:.1f} LU - it will sound weak next to other videos. "
                   "Raise gain (limiting may be needed).")
    else:
        out.append("Within 1 LU of target - leave it alone.")
    if tp is not None and tp > TP:
        out.append(f"True peak {tp} dBTP exceeds {TP} - risk of clipping after AAC encode.")
    out.append("Never re-master a label-supplied video master mix without approval; ask for a -14 LUFS video master instead.")
    return " ".join(out)


# ------------------------------------------------------------------ look / grade
def look_filter(w: int, h: int, lut: str | None = None, lut_strength: float = 1.0, preset: str = "none",
                exposure: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, temperature: float = 0.0,
                tint: float = 0.0, grain: float = 0.0, vignette: float = 0.0, letterbox: float | None = None,
                sharpen: float = 0.0) -> str:
    chain = []
    if lut:
        lp = str(Path(lut).expanduser().resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        if lut_strength >= 0.999:
            chain.append(f"lut3d=file='{lp}':interp=tetrahedral")
        else:
            chain.append(f"split[_a][_b];[_b]lut3d=file='{lp}':interp=tetrahedral[_l];"
                         f"[_a][_l]blend=all_mode=normal:all_opacity={lut_strength:.3f}")
    if exposure:
        chain.append(f"exposure=exposure={exposure:.3f}")
    if contrast != 1.0 or saturation != 1.0:
        chain.append(f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}")
    if temperature or tint:
        t = max(-1, min(1, temperature)) * 0.12
        g = max(-1, min(1, tint)) * 0.08
        chain.append(f"colorbalance=rs={t:.3f}:bs={-t:.3f}:rm={t:.3f}:bm={-t:.3f}:gm={-g:.3f}:gs={-g:.3f}")
    if preset and preset != "none":
        if preset not in LOOK_PRESETS:
            raise MediaError(f"Unknown look preset {preset}. Options: {list(LOOK_PRESETS)}")
        chain.append(LOOK_PRESETS[preset])
    if sharpen:
        chain.append(f"unsharp=5:5:{sharpen:.2f}")
    if vignette:
        chain.append(f"vignette=angle={0.2 + 0.5 * min(1, vignette):.3f}")
    if grain:
        chain.append(f"noise=c0s={int(4 + 20 * min(1, grain))}:c0f=t+u")
    if letterbox:
        target_h = w / letterbox
        bar = max(0, int((h - target_h) / 2))
        if bar > 0:
            chain.append(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,"
                         f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill")
    return ",".join(c for c in chain if c) or "null"


def apply_look(inp: str, output: str, preview_at: float | None = None, **kw) -> dict:
    info, v, a = _vinfo(inp)
    if not v:
        raise MediaError("No video stream")
    vf = look_filter(v["width"], v["height"], **kw)
    if preview_at is not None:
        # before/after split-screen still for review
        out = str(Path(output).with_suffix(".jpg"))
        fc = (f"[0:v]split[o][g];[g]{vf}[gr];[o]scale=960:-2[o2];[gr]scale=960:-2[g2];"
              f"[o2]drawtext=text='BEFORE':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[o3];"
              f"[g2]drawtext=text='AFTER':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[g3];"
              f"[o3][g3]hstack")
        ffmpeg(["-ss", str(preview_at), "-i", inp, "-frames:v", "1", "-filter_complex", fc, "-q:v", "3", out])
        return {"preview": out, "filter": vf, "note": "Look at this image before rendering the full clip."}
    ext = Path(output).suffix.lower()
    venc = (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] if ext == ".mov"
            else ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"])
    aargs = ["-c:a", "copy"] if a else []
    ffmpeg(["-i", inp, "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", *(["-map", "0:a:0"] if a else []),
            *venc, *REC709, *aargs, output])
    return {"output": output, "filter": vf}


# ------------------------------------------------------------------ lyrics
def parse_lyrics(text_or_path: str) -> list[dict]:
    p = Path(text_or_path).expanduser()
    raw = p.read_text(encoding="utf-8", errors="ignore") if (len(text_or_path) < 500 and p.exists()) else text_or_path
    lines: list[dict] = []
    if re.search(r"^\d+\s*\n\d\d:\d\d:\d\d,\d{3} -->", raw, re.M):   # SRT
        for blk in re.split(r"\n\s*\n", raw.strip()):
            m = re.search(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})\s*\n(.+)", blk, re.S)
            if m:
                g = [int(x) for x in m.groups()[:8]]
                lines.append({"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                              "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                              "text": " ".join(m.group(9).split())})
        return lines
    for ln in raw.splitlines():   # LRC: [mm:ss.xx] text  (multiple stamps allowed)
        stamps = re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", ln)
        txt = re.sub(r"\[[^\]]*\]", "", ln).strip()
        for mm, ss in stamps:
            lines.append({"start": int(mm) * 60 + float(ss), "end": None, "text": txt})
    lines.sort(key=lambda x: x["start"])
    for i, l in enumerate(lines):
        if l["end"] is None:
            nxt = lines[i + 1]["start"] if i + 1 < len(lines) else l["start"] + 4
            l["end"] = min(nxt, l["start"] + 6)
    return [l for l in lines if l["text"]]


def _ass_time(t: float) -> str:
    t = max(0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def write_ass(lines: list[dict], path: Path, w: int, h: int, font: str = "Inter", size_pct: float = 5.0,
              position: str = "bottom", color: str = "FFFFFF", outline: float = 2.5, uppercase: bool = False,
              offset: float = 0.0, fade_ms: int = 120) -> Path:
    align = {"bottom": 2, "center": 5, "top": 8, "lower_left": 1, "lower_right": 3}.get(position, 2)
    fs = int(h * size_pct / 100)
    c = color.lstrip("#").upper()
    bgr = f"&H00{c[4:6]}{c[2:4]}{c[0:2]}"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,{font},{fs},{bgr},&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,{outline},1,{align},{int(w*0.06)},{int(w*0.06)},{int(h*0.07)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for l in lines:
        txt = l["text"].upper() if uppercase else l["text"]
        txt = txt.replace("{", "(").replace("}", ")")
        ev.append(f"Dialogue: 0,{_ass_time(l['start'] - offset)},{_ass_time(l['end'] - offset)},Lyric,,0,0,0,,"
                  f"{{\\fad({fade_ms},{fade_ms})}}{txt}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")
    return path


def write_srt(lines: list[dict], path: Path, offset: float = 0.0) -> Path:
    def ts(t):
        ms = int(round(max(0, t) * 1000))
        return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d},{ms % 1000:03d}"
    out = [f"{i}\n{ts(l['start'] - offset)} --> {ts(l['end'] - offset)}\n{l['text']}\n" for i, l in enumerate(lines, 1)]
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def snap_lyrics(lines: list[dict], beats: list[float], tolerance: float = 0.12) -> list[dict]:
    if not beats:
        return lines
    b = np.array(beats)
    for l in lines:
        j = int(np.argmin(abs(b - l["start"])))
        if abs(b[j] - l["start"]) <= tolerance:
            l["start"] = float(b[j])
    return lines


def burn_subtitles(inp: str, ass: Path, output: str) -> None:
    sp = str(ass.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    info, v, a = _vinfo(inp)
    ext = Path(output).suffix.lower()
    venc = (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] if ext == ".mov"
            else ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"])
    ffmpeg(["-i", inp, "-vf", f"ass='{sp}'", *venc, *REC709, *(["-c:a", "copy"] if a else []), output])


# ------------------------------------------------------------------ deliverables
def render_preset(inp: str, output: str, preset: str, focus_x: float = 0.5, focus_y: float = 0.5,
                  start: float | None = None, length: float | None = None, watermark: str | None = None,
                  metadata: dict | None = None, fade: float = 0.0, no_audio: bool = False) -> dict:
    if preset not in DELIVERY_PRESETS:
        raise MediaError(f"Unknown preset {preset}. Options: {list(DELIVERY_PRESETS)}")
    P = DELIVERY_PRESETS[preset]
    info, v, a = _vinfo(inp)
    w, h = v["width"], v["height"]
    fps = fps_of(info) or 24
    warnings = []
    if P.get("height") and h < P["height"]:
        warnings.append(f"Source is {h}p; not upscaling to {P['height']}p - delivering at native {h}p.")
        height = None
    else:
        height = P.get("height")
    vf = _reframe(w, h, P.get("aspect"), P.get("size"), focus_x, focus_y, height)
    if P.get("review"):
        vf += (",drawtext=text='%{pts\\:hms}':x=w-tw-24:y=h-th-24:fontsize=h/28:fontcolor=white:box=1:boxcolor=black@0.5")
        if watermark:
            wm = watermark.replace("'", "").replace(":", " ")
            vf += f",drawtext=text='{wm}':x=(w-tw)/2:y=(h-th)/2:fontsize=h/12:fontcolor=white@0.22"
    elif watermark:
        wm = watermark.replace("'", "").replace(":", " ")
        vf += f",drawtext=text='{wm}':x=w-tw-30:y=30:fontsize=h/30:fontcolor=white@0.6"
    total = length or (duration(info) - (start or 0))
    if fade:
        vf += f",fade=t=in:st=0:d={fade},fade=t=out:st={max(0, total - fade):.3f}:d={fade}"
    pre = ["-ss", f"{start:.4f}"] if start else []
    dur = ["-t", f"{length:.4f}"] if length else []
    gop = ["-g", str(max(1, int(round(fps / 2))))] if "libx264" in P["v"] else []
    aargs = []
    if a and not no_audio:
        aargs = ["-map", "0:a:0", *P["a"]]
        if fade:
            aargs += ["-af", f"afade=t=in:st=0:d={min(fade, 0.1)},afade=t=out:st={max(0, total - fade):.3f}:d={fade}"]
    else:
        aargs = ["-an"]
    meta = []
    for k, val in (metadata or {}).items():
        meta += ["-metadata", f"{k}={val}"]
    mov = ["-movflags", "+faststart"] if P["ext"] in ("mp4", "mov") else []
    ffmpeg([*pre, "-i", inp, *dur, "-map", "0:v:0", "-vf", vf, *P["v"], *gop, *REC709, *aargs, *meta, *mov, output])
    out_info = probe(output)
    ov = video_stream(out_info)
    return {"preset": preset, "output": output, "desc": P["desc"], "resolution": f"{ov['width']}x{ov['height']}",
            "duration_s": round(duration(out_info), 2), "size_mb": round(Path(output).stat().st_size / 1e6, 1),
            "warnings": warnings}


def best_window(analysis: dict, length_s: float, section: str = "auto") -> tuple[float, float, str]:
    """Pick a start time on a bar line for a cutdown of ~length_s."""
    bar = analysis["bar_length_s"]
    n_bars = max(1, round(length_s / bar))
    secs = analysis["sections"]
    if section == "auto":
        choruses = [s for s in secs if s.get("name") == "chorus"] or sorted(secs, key=lambda s: -s["energy"])[:1]
        target = max(choruses, key=lambda s: s["energy"])
        why = f"highest-energy {target.get('name', 'section')} at {target['start']:.1f}s"
    else:
        cands = [s for s in secs if s.get("name") == section or s["label"] == section]
        if not cands:
            raise MediaError(f"No section named {section}. Have: {[s.get('name') for s in secs]}")
        target = cands[0]
        why = f"first {section}"
    # start 1 bar before the hook so it lands on the downbeat with a pickup
    bars = analysis["bar_starts"]
    idx = min(range(len(bars)), key=lambda i: abs(bars[i] - target["start"]))
    start_idx = max(0, idx - 1)
    start = bars[start_idx]
    end_idx = min(len(bars) - 1, start_idx + n_bars)
    end = bars[end_idx] if end_idx > start_idx and start_idx + n_bars < len(bars) else min(analysis["duration"], start + n_bars * bar)
    return start, end, why + f"; 1-bar pickup, {n_bars} bars"


# ------------------------------------------------------------------ conform / proxies
CONFORM_CODECS = {
    "prores_proxy": ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"],
    "prores_lt": ["-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le"],
    "prores_422": ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"],
    "prores_hq": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    "dnxhr_hq": ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"],
    "h264_edit": ["-c:v", "libx264", "-preset", "fast", "-crf", "14", "-g", "1", "-pix_fmt", "yuv420p"],
}


def conform(inp: str, output: str, fps: float, playback_speed: float = 1.0, codec: str = "prores_422",
            max_height: int | None = None) -> dict:
    info, v, a = _vinfo(inp)
    if codec not in CONFORM_CODECS:
        raise MediaError(f"codec options: {list(CONFORM_CODECS)}")
    src_fps = fps_of(info) or fps
    vf = []
    if playback_speed != 1.0:
        vf.append(f"setpts=PTS*{playback_speed:.6f}")
    vf.append(f"fps={fps:.6f}")
    if max_height and v["height"] > max_height:
        vf.append(f"scale=-2:{max_height}")
    keep_audio = a is not None and playback_speed == 1.0
    ffmpeg(["-i", inp, "-map", "0:v:0", *(["-map", "0:a:0", "-c:a", "pcm_s24le", "-ar", "48000"] if keep_audio else ["-an"]),
            "-vf", ",".join(vf), "-fps_mode", "cfr", *CONFORM_CODECS[codec], "-map_metadata", "0", output])
    return {"output": output, "source_fps": round(src_fps, 3), "output_fps": fps, "playback_speed": playback_speed,
            "slowdown": f"{src_fps / fps:.2f}x frames-per-second ratio" if playback_speed != 1 else None,
            "audio": "kept (48k PCM)" if keep_audio else "dropped (scratch audio is meaningless after a speed conform)",
            "duration_s": round(duration(probe(output)), 3)}


# ------------------------------------------------------------------ frames / thumbnails / storyboard
def grab_frame(inp: str, t: float, output: str, width: int = 1280) -> str:
    ffmpeg(["-ss", f"{max(0, t):.3f}", "-i", inp, "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", output])
    return output


def _gray(inp: str, t: float, w: int = 320) -> np.ndarray | None:
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", inp, "-frames:v", "1",
                        "-vf", f"scale={w}:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    if p.returncode or not p.stdout:
        return None
    a = np.frombuffer(p.stdout, np.uint8)
    hgt = a.size // w
    return a[: hgt * w].reshape(hgt, w).astype(np.float32) if hgt else None


def score_frame(g: np.ndarray) -> dict:
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    sharp = float(lap.var())
    mean = float(g.mean())
    contrast = float(g.std())
    expo_pen = math.exp(-((mean - 115) / 60) ** 2)
    return {"sharpness": sharp, "brightness": mean, "contrast": contrast,
            "score": math.log1p(sharp) * expo_pen * (0.5 + min(contrast, 60) / 60)}


def contact_sheet(images: list[str], output: str, cols: int = 4, labels: list[str] | None = None,
                  tile_w: int = 480) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="mvd_sheet_"))
    try:
        for i, im in enumerate(images):
            lab = (labels[i] if labels else "").replace("'", "").replace(":", "\\:")
            vf = f"scale={tile_w}:-2,pad={tile_w}:ceil(ih/2)*2+0"
            if lab:
                vf += f",drawtext=text='{lab}':x=8:y=h-th-8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
            ffmpeg(["-i", im, "-vf", vf, "-frames:v", "1", str(tmp / f"{i:04d}.png")])
        rows = math.ceil(len(images) / cols)
        ffmpeg(["-framerate", "1", "-i", str(tmp / "%04d.png"), "-vf", f"tile={cols}x{rows}:padding=4:color=black",
                "-frames:v", "1", "-q:v", "3", output])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return output


# ------------------------------------------------------------------ QC
QC_SPECS = {
    "youtube": {"lufs": (-14, 3), "tp": -1.0, "codecs": None},
    "streaming_social": {"lufs": (-14, 2), "tp": -1.0, "codecs": ["h264", "hevc"]},
    "label_master": {"lufs": None, "tp": -1.0, "codecs": ["prores"]},
    "broadcast_ebu": {"lufs": (-23, 1), "tp": -1.0, "codecs": None},
}


def qc(inp: str, spec: str = "youtube", master_song: str | None = None, song_offset: float = 0.0) -> dict:
    from .analysis import sync_offset
    from .media import summarize
    S = QC_SPECS.get(spec, QC_SPECS["youtube"])
    summ = summarize(inp)
    items: list[dict] = []

    def add(level, check, detail):
        items.append({"level": level, "check": check, "detail": detail})
    info = probe(inp)
    v, a = video_stream(info), audio_stream(info)
    fps = fps_of(info) or 24
    for w in summ["warnings"]:
        add("WARN", "media", w)
    if v:
        vv = summ["video"]
        if S["codecs"] and vv["codec"] not in S["codecs"]:
            add("FAIL", "codec", f"{vv['codec']} not in {S['codecs']} for {spec}")
        if vv["pix_fmt"] not in ("yuv420p", "yuv422p10le", "yuv420p10le", "yuv444p10le", "yuva444p10le", "yuv422p"):
            add("WARN", "pixel format", f"{vv['pix_fmt']} - some players/sites mis-handle it")
        if vv["codec"] == "h264" and vv["pix_fmt"] != "yuv420p":
            add("FAIL", "pixel format", "H.264 must be yuv420p for web/phone playback")
        if not vv["color_primaries"] or vv["color_primaries"] == "unknown" or not vv["color_transfer"]:
            add("WARN", "colour tags", "No BT.709 colour tags - QuickTime/YouTube may shift gamma (washed-out look). "
                "Re-render with render_deliverables (it tags BT.709).")
        std = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
        if min(abs(fps - s) for s in std) > 0.01:
            add("WARN", "frame rate", f"Non-standard {fps:.3f} fps")
        if vv["width"] % 2 or vv["height"] % 2:
            add("FAIL", "dimensions", "Odd width/height")
    else:
        add("FAIL", "video", "No video stream")
    if not a:
        add("FAIL", "audio", "No audio stream")
    # one decode pass for picture/sound faults
    p = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", inp,
                        "-vf", "blackdetect=d=0.25:pix_th=0.08,freezedetect=n=0.002:d=2",
                        "-af", "silencedetect=n=-55dB:d=1.5,astats=metadata=0:reset=0",
                        "-f", "null", "-"], capture_output=True, text=True)
    err = p.stderr
    blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)
    dur = duration(info)
    for s_, e_ in blacks:
        s_, e_ = float(s_), float(e_)
        edge = s_ < 0.05 or e_ > dur - 0.1
        add("INFO" if edge else "WARN", "black frames",
            f"{seconds_to_tc(s_, fps)} - {seconds_to_tc(e_, fps)} ({e_ - s_:.2f}s){' at head/tail (fine for fades)' if edge else ' mid-programme - flash frame or gap?'}")
    for fs in re.findall(r"freeze_start: ([\d.]+)", err):
        add("WARN", "frozen picture", f"from {seconds_to_tc(float(fs), fps)} (>2s identical frames - render error or hold?)")
    for ss in re.findall(r"silence_start: ([\d.]+)", err):
        t = float(ss)
        add("INFO" if t > dur - 3 or t < 0.5 else "WARN", "silence", f"from {seconds_to_tc(t, fps)}")
    pk = re.findall(r"Peak level dB: (-?[\d.inf]+)", err)
    if pk:
        try:
            peak = max(float(x) for x in pk if "inf" not in x)
            if peak >= -0.1:
                add("FAIL", "clipping", f"Sample peak {peak} dBFS - digital overs")
        except ValueError:
            pass
    if a:
        L = loudness(inp)
        if S["lufs"] and L["integrated_lufs"] is not None:
            tgt, tol = S["lufs"]
            d = L["integrated_lufs"] - tgt
            add("PASS" if abs(d) <= tol else "WARN", "loudness",
                f"{L['integrated_lufs']} LUFS (target {tgt} +/-{tol})")
        if L["true_peak_dbtp"] is not None:
            add("PASS" if L["true_peak_dbtp"] <= S["tp"] else "WARN", "true peak",
                f"{L['true_peak_dbtp']} dBTP (max {S['tp']})")
        vd = float(v.get("duration") or dur) if v else dur
        ad = float(a.get("duration") or dur)
        if v and abs(vd - ad) > 0.1:
            add("WARN", "stream lengths", f"Video {vd:.2f}s vs audio {ad:.2f}s")
    if master_song and a:
        r = sync_offset(master_song, inp)
        if r.get("ok"):
            err_s = r["offset_s"] - song_offset
            fr = err_s * fps
            add("PASS" if abs(fr) < 0.6 else "FAIL", "master audio sync",
                f"Audio is {err_s * 1000:+.0f} ms ({fr:+.1f} frames) vs master (confidence {r['confidence']})")
    levels = [i["level"] for i in items]
    verdict = "FAIL" if "FAIL" in levels else ("PASS WITH WARNINGS" if "WARN" in levels else "PASS")
    return {"file": inp, "spec": spec, "verdict": verdict, "checks": items, "summary": summ}
