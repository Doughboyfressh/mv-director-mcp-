"""Beat-synced effects: Flash, Optical Shake, Zoom (punch-out), Smooth Rotation, Smooth Slide (whip),
Skipping Frames (stutter) and Night Vision. Placement comes from the song analysis (beats, downbeats,
sections, impact moment) and from cuts detected in the video; rendering is one ffmpeg filtergraph."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .media import FFMPEG, MediaError, audio_stream, duration, ffmpeg, fps_of, probe, seconds_to_tc, video_stream

POINT_FX = ("flash", "shake", "zoom", "rotation", "slide")
WINDOW_FX = ("skip", "nightvision")
TRANSITIONS = ("zoom", "rotation", "slide")
ALIASES = {"zoom_out": "zoom", "zoomout": "zoom", "optical_shake": "shake", "smooth_rotation": "rotation",
           "smooth_slide": "slide", "whip": "slide", "wlr": "slide", "skipping_frames": "skip", "stutter": "skip",
           "night_vision": "nightvision", "flash": "flash"}

STYLES = {
    "hype": [
        {"type": "flash", "where": "impact", "intensity": 1.0},
        {"type": "flash", "where": "chorus_starts", "intensity": 0.8},
        {"type": "flash", "where": "chorus_downbeats", "intensity": 0.45, "every": 2},
        {"type": "shake", "where": "impact", "intensity": 1.0},
        {"type": "shake", "where": "chorus_beats", "intensity": 0.35},
        {"type": "transitions", "where": "cuts_on_beat", "intensity": 0.8, "min_gap_bars": 1},
        {"type": "skip", "where": "build_ups", "intensity": 0.8},
    ],
    "smooth": [
        {"type": "flash", "where": "impact", "intensity": 0.5},
        {"type": "shake", "where": "impact", "intensity": 0.5},
        {"type": "transitions", "where": "cuts_on_beat", "intensity": 0.55, "min_gap_bars": 2},
    ],
    "subtle": [
        {"type": "flash", "where": "impact", "intensity": 0.35},
        {"type": "shake", "where": "impact", "intensity": 0.3},
        {"type": "zoom", "where": "section_starts", "intensity": 0.3},
    ],
}
STYLE_DESC = {
    "hype": "Rap/drill/EDM energy: flashes on the drop and chorus downbeats, shake on the beat in hooks, "
            "zoom/slide/rotation transitions on on-beat cuts, stutter on the bar before each chorus.",
    "smooth": "R&B/pop: transitions every couple of bars on on-beat cuts, one flash + shake on the biggest lift.",
    "subtle": "Cinematic/ballad: a light flash and shake on the impact, gentle punch-outs at section changes.",
}


# ------------------------------------------------------------------ cut detection
def detect_cuts(video: str, threshold: float = 10.0) -> list[float]:
    """Hard cuts via scene-change detection (threshold in %, lower = more sensitive)."""
    p = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", video, "-an", "-vf",
                        f"scale=320:-2,scdet=threshold={threshold}:sc_pass=0", "-f", "null", "-"],
                       capture_output=True, text=True)
    times = [float(x) for x in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", p.stderr)]
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] > 0.15:
            out.append(round(t, 3))
    return out


def cut_report(cuts: list[float], analysis: dict | None, offset: float, fps: float, video_dur: float) -> dict:
    rep = {"cuts": len(cuts), "asl_s": round(video_dur / (len(cuts) + 1), 2),
           "cut_times": [{"t": c, "tc": seconds_to_tc(c, fps)} for c in cuts[:300]]}
    if analysis and cuts:
        beats = np.array(analysis["beats"]) - offset
        downs = np.array(analysis["downbeats"]) - offset
        d_beat = np.array([np.min(np.abs(beats - c)) for c in cuts]) * fps
        d_down = np.array([np.min(np.abs(downs - c)) for c in cuts]) * fps
        for i, c in enumerate(rep["cut_times"]):
            c["frames_off_beat"] = round(float(d_beat[i]), 1)
            c["on_downbeat"] = bool(d_down[i] <= 1.5)
        rep["on_beat_within_1_frame_pct"] = round(100 * float(np.mean(d_beat <= 1.0)), 1)
        rep["on_beat_within_2_frames_pct"] = round(100 * float(np.mean(d_beat <= 2.0)), 1)
        rep["on_downbeat_pct"] = round(100 * float(np.mean(d_down <= 1.5)), 1)
        off = [c for c in rep["cut_times"] if c["frames_off_beat"] > 2]
        rep["off_beat_cuts"] = off[:40]
        pct = rep["on_beat_within_2_frames_pct"]
        rep["verdict"] = ("Tight - the edit is locked to the beat." if pct >= 80 else
                          "Mostly on beat - check the listed off-beat cuts." if pct >= 55 else
                          "Loose - many cuts drift off the beat (fine if intentional, e.g. cutting on lyrics).")
    return rep


# ------------------------------------------------------------------ placement
def _sec(analysis, name):
    return [s for s in analysis["sections"] if s.get("name") == name or s["label"] == name]


def _hi_energy_bars(analysis):
    be = np.array(analysis["bar_energy"] or [0])
    thr = np.percentile(be, 70) if be.size else 0
    bars = analysis["bar_starts"] + [analysis["duration"]]
    return [(bars[i], bars[i + 1]) for i in range(len(be)) if be[i] >= thr]


def _points(where, analysis, cuts, fps, every: int = 1) -> list[float]:
    a = analysis
    if isinstance(where, (list, tuple)):
        return [float(x) for x in where]
    w = str(where)
    if w == "impact":
        return [a["impact_moment"]["time"]] if a.get("impact_moment") else []
    if w == "downbeats":
        pts = list(a["downbeats"])
    elif w == "beats":
        pts = list(a["beats"])
    elif w in ("chorus_downbeats", "chorus_beats"):
        ch = _sec(a, "chorus")
        src = a["downbeats"] if w == "chorus_downbeats" else a["beats"]
        pts = [t for t in src if any(s["start"] - 0.02 <= t < s["end"] for s in ch)]
        if w == "chorus_beats":  # beats 1 and 3 only
            first = a["downbeats"][0] if a["downbeats"] else 0
            bi = a["beat_interval_s"]
            pts = [t for t in pts if round((t - first) / bi) % 2 == 0]
    elif w == "chorus_starts":
        pts = [s["start"] for s in _sec(a, "chorus")]
    elif w == "section_starts":
        pts = [s["start"] for s in a["sections"][1:]]
    elif w == "hi_energy_beats":
        hb = _hi_energy_bars(a)
        pts = [t for t in a["beats"] if any(x <= t < y for x, y in hb)]
    elif w in ("cuts", "cuts_on_beat", "chorus_cuts"):
        pts = list(cuts)
        if w == "cuts_on_beat":
            beats = np.array(a["beats"])
            pts = [c for c in pts if beats.size and np.min(np.abs(beats - c)) * fps <= 2.0]
        if w == "chorus_cuts":
            pts = [c for c in pts if any(s["start"] <= c < s["end"] for s in _sec(a, "chorus"))]
    elif w.startswith("section:"):
        pts = [s["start"] for s in _sec(a, w.split(":", 1)[1])]
    else:
        raise MediaError(f"Unknown placement '{w}'. Use impact, downbeats, beats, chorus_downbeats, chorus_beats, "
                         "chorus_starts, section_starts, hi_energy_beats, cuts, cuts_on_beat, chorus_cuts, "
                         "section:<name>, or a list of seconds.")
    return pts[::max(1, every)]


def _windows(where, analysis) -> list[tuple[float, float]]:
    a = analysis
    bar = a["bar_length_s"]
    if isinstance(where, (list, tuple)):
        out = []
        for x in where:
            if isinstance(x, (list, tuple)):
                out.append((float(x[0]), float(x[1])))
            else:
                out.append((float(x), float(x) + a["beat_interval_s"] * 2))
        return out
    w = str(where)
    if w == "build_ups":
        return [(max(0, s["start"] - bar), s["start"]) for s in _sec(a, "chorus")]
    if w.startswith("section:"):
        return [(s["start"], s["end"]) for s in _sec(a, w.split(":", 1)[1])]
    if w.startswith("range:"):
        x, y = w.split(":", 1)[1].split("-")
        return [(float(x), float(y))]
    if w == "chorus":
        return [(s["start"], s["end"]) for s in _sec(a, "chorus")]
    if w == "impact":
        t = (a.get("impact_moment") or {}).get("time")
        return [(t, t + bar)] if t is not None else []
    raise MediaError(f"Unknown window placement '{w}'. Use build_ups, chorus, impact, section:<name>, "
                     "range:<start>-<end>, or a list of [start, end] pairs.")


def plan(analysis: dict, cuts_video: list[float], offset: float, fps: float, video_dur: float,
         style: str | None = "hype", effects: list[dict] | None = None) -> list[dict]:
    """Return events in VIDEO time: {type, t, intensity} or {type, start, end, intensity}."""
    if style and style not in STYLES:
        raise MediaError(f"style must be one of {list(STYLES)}")
    recipe = list(STYLES.get(style, [])) if style else []
    recipe += effects or []
    cuts_song = [c + offset for c in cuts_video]
    bar = analysis["bar_length_s"]
    ev: list[dict] = []
    cycle = 0
    for r in recipe:
        typ = ALIASES.get(str(r["type"]).lower(), str(r["type"]).lower())
        inten = float(r.get("intensity", 0.7))
        where = r.get("where", "impact")
        if typ == "transitions" or (typ in TRANSITIONS and where in ("cuts", "cuts_on_beat", "chorus_cuts")
                                    and r.get("cycle", typ == "transitions")):
            pts = _points(where, analysis, cuts_song, fps, int(r.get("every", 1)))
            if not pts and isinstance(where, str) and where.startswith("cuts"):  # one-take video: use chorus downbeats every 2 bars
                pts = _points("chorus_downbeats", analysis, cuts_song, fps, 2)
            gap = float(r.get("min_gap_bars", 1)) * bar
            last = -1e9
            kinds = r.get("kinds", list(TRANSITIONS))
            for t in pts:
                if t - last >= gap - 0.05:
                    ev.append({"type": kinds[cycle % len(kinds)] if typ == "transitions" else typ,
                               "t": t, "intensity": inten})
                    cycle += 1
                    last = t
        elif typ in POINT_FX:
            for t in _points(where, analysis, cuts_song, fps, int(r.get("every", 1))):
                ev.append({"type": typ, "t": t, "intensity": inten})
        elif typ in WINDOW_FX:
            for s, e in _windows(where, analysis):
                ev.append({"type": typ, "start": s, "end": e, "intensity": inten})
        else:
            raise MediaError(f"Unknown effect '{r['type']}'. Effects: flash, shake, zoom, rotation, slide, skip, "
                             "nightvision, transitions")
    # song time -> video time, clip to video, de-duplicate
    out, seen = [], {}
    min_gap = {"flash": 0.2, "shake": 0.2, "zoom": 0.5, "rotation": 0.5, "slide": 0.5}
    for e in sorted(ev, key=lambda x: (x.get("t", x.get("start")), -x["intensity"])):
        if "t" in e:
            t = e["t"] - offset
            if not (0.05 < t < video_dur - 0.05):
                continue
            lane = "transition" if e["type"] in TRANSITIONS else e["type"]
            if t - seen.get(lane, -1e9) < min_gap.get(e["type"], 0.2):
                continue
            seen[lane] = t
            out.append({**e, "t": round(t, 3), "tc": seconds_to_tc(t, fps)})
        else:
            s, en = max(0.0, e["start"] - offset), min(video_dur, e["end"] - offset)
            if en - s > 0.05:
                out.append({**e, "start": round(s, 3), "end": round(en, 3), "tc": seconds_to_tc(s, fps)})
    return out


# ------------------------------------------------------------------ filtergraph
def _sum(terms: list[str]) -> str:
    return "+".join(terms) if terms else "0"


def _win(T: str, t: float, pre: float, post: float, body: str) -> str:
    return f"if(between({T},{t - pre:.4f},{t + post:.4f}),{body},0)"


def _enable(wins: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in wins) or "0"


def build_graph(events: list[dict], W: int, H: int, fps: float) -> str:
    by = {k: [e for e in events if e["type"] == k] for k in POINT_FX + WINDOW_FX}
    chain: list[str] = []
    label = "0:v"
    n = 0

    def step(f: str) -> None:
        nonlocal label, n
        n += 1
        chain.append(f"[{label}]{f}[v{n}]")
        label = f"v{n}"

    # Night vision (windowed colour treatment)
    if by["nightvision"]:
        en = _enable([(e["start"], e["end"]) for e in by["nightvision"]])
        g = max(e["intensity"] for e in by["nightvision"])
        step(f"hue=s=0:enable='{en}',"
             f"colorchannelmixer=rr=0.1:rg=0.25:rb=0.05:gr=0.25:gg={1.0 + 0.3 * g:.2f}:gb=0.15:br=0.05:bg=0.2:bb=0.05:enable='{en}',"
             f"eq=contrast={1.1 + 0.3 * g:.2f}:brightness=0.04:enable='{en}',"
             f"noise=alls={int(12 + 28 * g)}:allf=t:enable='{en}',vignette=angle=PI/3.2:enable='{en}'")

    # Smooth Rotation: spin through the cut (angle decays to 0), corners hidden by the zoom stage
    rot_terms = []
    for e in by["rotation"]:
        A = 0.22 * e["intensity"] * (1 if (len(rot_terms) % 2 == 0) else -1)
        d = f"(t-{e['t']:.4f})"
        rot_terms.append(_win("t", e["t"], 0.35, 1.2, f"{A:.4f}*if(lt({d},0),-exp(-pow({d}/0.09,2)),exp(-{d}/0.17))"))
    if rot_terms:
        step(f"rotate=a='{_sum(rot_terms)}':c=black:ow=iw:oh=ih")

    # Zoom: punch-out (peaks on the cut, eases back to 1.0) + rotation corner cover
    z_terms = []
    for e in by["zoom"]:
        A = 0.4 * e["intensity"]
        d = f"(it-{e['t']:.4f})"
        z_terms.append(_win("it", e["t"], 0.35, 1.6, f"{A:.4f}*if(lt({d},0),exp(-pow({d}/0.07,2)),exp(-{d}/0.22))"))
    for e in by["rotation"]:
        A = 0.22 * e["intensity"] * (W / H + 0.45)
        d = f"(it-{e['t']:.4f})"
        z_terms.append(_win("it", e["t"], 0.35, 1.2, f"{A:.4f}*if(lt({d},0),exp(-pow({d}/0.09,2)),exp(-{d}/0.17))"))
    if z_terms:
        step(f"zoompan=z='1+{_sum(z_terms)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={fps:.6f}")

    # Smooth Slide / whip: frame wraps horizontally through one full width around the cut
    sl_terms = []
    for i, e in enumerate(by["slide"]):
        sgn = 1 if i % 2 == 0 else -1
        span = 0.26 / max(0.4, e["intensity"])
        s = f"clip((t-{e['t'] - span / 2:.4f})/{span:.4f},0,1)"
        sl_terms.append(f"{sgn * W}*({s})*({s})*(3-2*({s}))")
    if sl_terms:
        n += 1
        chain.append(f"[{label}]split[sa{n}][sb{n}];[sa{n}][sb{n}]hstack[sh{n}]")
        label = f"sh{n}"
        step(f"crop=w={W}:h={H}:x='mod({_sum(sl_terms)},{W})':y=0")

    # Optical Shake: decaying jitter inside a slightly enlarged frame
    sx, sy = [], []
    for e in by["shake"]:
        A = 0.032 * e["intensity"] * W
        d = f"(t-{e['t']:.4f})"
        sx.append(_win("t", e["t"], 0.01, 1.5, f"{A:.2f}*exp(-{d}/0.28)*sin(81.7*{d})"))
        sy.append(_win("t", e["t"], 0.01, 1.5, f"{A * 0.8:.2f}*exp(-{d}/0.28)*cos(69.1*{d}+1)"))
    if sx:
        m = int(W * 0.08) // 2 * 2
        mh = int(H * 0.08) // 2 * 2
        step(f"scale={W + m}:{H + mh},crop=w={W}:h={H}:x='{m // 2}+clip({_sum(sx)},-{m // 2},{m // 2})'"
             f":y='{mh // 2}+clip({_sum(sy)},-{mh // 2},{mh // 2})'")

    # Flash: white pop decaying over ~0.1 s
    fl = [_win("t", e["t"], 0.0, 0.6, f"{0.6 * e['intensity']:.3f}*exp(-(t-{e['t']:.4f})/0.085)") for e in by["flash"]]
    if fl:
        step(f"eq=brightness='min(0.85,{_sum(fl)})':contrast='1-0.25*min(1,{_sum(fl)})':eval=frame")

    # Skipping Frames: drop to a choppy frame rate inside the windows
    if by["skip"]:
        g = max(e["intensity"] for e in by["skip"])
        low = max(4, int(round(fps / (2 + 3 * g))))
        en = _enable([(e["start"], e["end"]) for e in by["skip"]])
        n += 1
        chain.append(f"[{label}]split[ka{n}][kb{n}];[kb{n}]fps={low},fps={fps:.6f}[kc{n}];"
                     f"[ka{n}][kc{n}]overlay=enable='{en}':eof_action=pass[v{n}]")
        label = f"v{n}"

    # Motion blur during whips and spins
    blur = [(e["t"] - 0.12, e["t"] + 0.18) for e in by["slide"] + by["rotation"]]
    if blur:  # blurred copy layered in only during the moves (keeps the main picture frame-exact)
        n += 1
        chain.append(f"[{label}]split[ba{n}][bb{n}];[bb{n}]tmix=frames=3[bt{n}];"
                     f"[ba{n}][bt{n}]overlay=enable='{_enable(blur)}':eof_action=pass[v{n}]")
        label = f"v{n}"

    step("format=yuv420p")
    return ";".join(chain), label


def render(video: str, output: str, events: list[dict], start: float | None = None,
           length: float | None = None) -> dict:
    info = probe(video)
    v, a = video_stream(info), audio_stream(info)
    if not v:
        raise MediaError("No video stream")
    W, H = v["width"] // 2 * 2, v["height"] // 2 * 2
    fps = fps_of(info) or 24.0
    evs = events
    if start:
        evs = []
        for e in events:
            e2 = dict(e)
            for k in ("t", "start", "end"):
                if k in e2:
                    e2[k] = e2[k] - start
            evs.append(e2)
    graph, out_label = build_graph(evs, W, H, fps)
    if W != v["width"] or H != v["height"]:
        graph = graph.replace("[0:v]", "[0:v]crop=%d:%d,setsar=1," % (W, H), 1) if graph.startswith("[0:v]") else graph
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(graph)
        script = f.name
    ext = Path(output).suffix.lower()
    venc = (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] if ext == ".mov" else
            ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    args = []
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", video]
    if length:
        args += ["-t", f"{length:.3f}"]
    args += ["-filter_complex_script", script, "-map", f"[{out_label}]"]
    if a:
        args += ["-map", "0:a:0", "-c:a", "aac" if ext != ".mov" else "pcm_s24le", "-b:a", "320k"]
    args += [*venc, "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", output]
    try:
        ffmpeg(args)
    finally:
        Path(script).unlink(missing_ok=True)
    return {"output": output, "duration_s": round(duration(probe(output)), 2)}
