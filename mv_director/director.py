"""Editorial brain: pacing, cut placement on the musical grid, take selection, coverage."""
from __future__ import annotations

import re

import numpy as np
from pathlib import Path

from .media import seconds_to_tc

# shot length in beats for (low, mid, high) energy
STYLES = {
    "performance": {"beats": (8, 4, 2), "broll": 0.25, "desc": "Artist-forward. Lip-sync is king; cut on downbeats, faster in hooks."},
    "hype": {"beats": (4, 2, 1), "broll": 0.35, "desc": "Rap/EDM energy. Cuts on every beat in drops, flash frames welcome."},
    "cinematic": {"beats": (16, 8, 4), "broll": 0.5, "desc": "Lets shots breathe. Long takes, motivated moves, cut on phrase ends."},
    "ballad": {"beats": (16, 8, 8), "broll": 0.4, "desc": "Emotional. Hold on faces; slow push-ins; cut on bar lines, never on beat 2/4."},
    "narrative": {"beats": (8, 8, 4), "broll": 0.6, "desc": "Story-led. Story scenes in verses, performance anchors the hooks."},
}

SIZE_TOKENS = [
    ("ECU", r"\b(ecu|xcu|extreme.?close|macro)\b"), ("CU", r"\b(cu|close|closeup|close.?up|tight)\b"),
    ("MS", r"\b(ms|mcu|med|medium|mid)\b"), ("WS", r"\b(ws|wide|full|ews|establish\w*|master)\b"),
    ("AERIAL", r"\b(drone|aerial|fpv)\b"),
]
MOVE_TOKENS = [("gimbal", r"gimbal|ronin|steadi"), ("handheld", r"handheld|hh\b"),
               ("dolly", r"dolly|slider|track"), ("slowmo", r"slow|slomo|overcrank|\d+x\b|120|240")]

SECTION_NOTES = {
    "intro": "Set the world. Establishing/aerial/texture shots, title card space. Let it breathe; hold the first performance face until the vocal enters.",
    "verse": "Story + medium performance. Favour MS/CU on the lead vocal for lip-sync lines; B-roll between phrases, not across them.",
    "pre-chorus": "Build tension: tighter framing each bar, cuts accelerate toward the chorus downbeat. A whip pan or speed-ramp into the hook lands well here.",
    "chorus": "Pay-off. Widest, best-lit, highest-production shots; cut ON the downbeat of bar 1. Alternate sizes (WS > CU > MS) so cuts never jump.",
    "bridge": "Change the visual language: new location, colour shift, or single long take. Audiences notice the switch subconsciously.",
    "breakdown": "Strip back: slow-mo, silhouettes, negative space. Fewer cuts make the next drop hit harder.",
    "outro": "Resolve. Pull wide, slow fade, or callback to the opening frame. Leave room for end card/credits.",
}


def shot_size(name: str) -> str:
    n = Path(name).stem.lower().replace("_", " ").replace("-", " ")
    for size, pat in SIZE_TOKENS:
        if re.search(pat, n):
            return size
    return "?"


def shot_move(name: str) -> str | None:
    n = Path(name).stem.lower()
    for mv, pat in MOVE_TOKENS:
        if re.search(pat, n):
            return mv
    return None


def _section_at(analysis: dict, t: float) -> dict:
    for s in analysis["sections"]:
        if s["start"] <= t < s["end"]:
            return s
    return analysis["sections"][-1]


def _energy_at(analysis: dict, t: float) -> float:
    bars = analysis["bar_starts"]
    e = analysis["bar_energy"]
    idx = max(0, min(len(e) - 1, sum(1 for b in bars if b <= t) - 1))
    return e[idx]


def coverage(analysis: dict, clips: list[dict]) -> list[dict]:
    """Per-section list of which takes cover it (song-time ranges from sync offsets)."""
    out = []
    for s in analysis["sections"]:
        cov = []
        for c in clips:
            if not c.get("sync", {}).get("ok"):
                continue
            a = c["song_in"]
            b = c["song_out"]
            ov = max(0.0, min(b, s["end"]) - max(a, s["start"]))
            if ov > 0:
                cov.append({"clip": Path(c["path"]).name, "role": c.get("role", "performance"),
                            "covered_pct": round(100 * ov / (s["end"] - s["start"]))})
        perf = [x for x in cov if x["role"] == "performance"]
        flag = None
        if not perf:
            flag = "NO PERFORMANCE COVERAGE - need B-roll or a pickup shot here"
        elif len(cov) == 1:
            flag = "Single take only - no cutting options (one long shot, or add B-roll/punch-ins)"
        out.append({"section": s.get("name", s["label"]), "start": s["start"], "end": s["end"],
                    "takes": cov, "performance_takes": len(perf), "flag": flag})
    return out


def plan_cuts(analysis: dict, clips: list[dict], fps: float, style: str = "performance", pace: float = 1.0,
              min_shot_frames: int = 12, broll_ratio: float | None = None, start_tc: str = "01:00:00:00") -> dict:
    st = STYLES.get(style, STYLES["performance"])
    broll_ratio = st["broll"] if broll_ratio is None else broll_ratio
    beats = analysis["beats"]
    bpb = analysis["beats_per_bar"]
    dur = analysis["duration"]
    usable = [c for c in clips if c.get("sync", {}).get("ok")]
    impact = (analysis.get("impact_moment") or {}).get("time")

    # Work on the beat grid. Shots >= 1 bar are snapped to bar lines; section starts + impact are forced cuts.
    first_db = analysis["downbeats"][0] if analysis["downbeats"] else (beats[0] if beats else 0.0)
    phase = min(range(len(beats)), key=lambda j: abs(beats[j] - first_db)) % bpb if beats else 0
    forced = sorted({round(s["start"], 3) for s in analysis["sections"]} | ({round(impact, 3)} if impact else set()))

    def bidx(t: float) -> int:
        return min(range(len(beats)), key=lambda j: abs(beats[j] - t))

    be = analysis["bar_energy"] or [0.5]
    lo_thr, hi_thr = float(np.percentile(be, 35)), float(np.percentile(be, 75))
    cuts_t: list[float] = [0.0]
    t = 0.0
    k = 0
    while beats and t < dur - 0.05:
        e = _energy_at(analysis, t)
        lvl = 0 if e < lo_thr else (1 if e < hi_thr else 2)
        n_beats = max(1, round(st["beats"][lvl] / pace))
        if k % 4 == 3 and lvl < 2:
            n_beats *= 2  # rhythmic variation: every 4th shot holds longer
        i0 = bidx(t) if t > 0 else 0
        if t == 0 and beats[0] > 0.2:   # pre-roll before the first beat: first cut on first downbeat
            i0 = phase - bpb if phase else 0
        j = i0 + n_beats
        if n_beats >= bpb:
            j = j - ((j - phase) % bpb)   # land on a bar line
            if j <= i0:
                j += bpb
        nxt = beats[j] if 0 <= j < len(beats) else dur
        f_in = [f for f in forced if t + 1e-3 < f < nxt - 1e-3]
        if f_in:
            nxt, k = f_in[0], 0
        else:
            k += 1
        if (nxt - t) * fps < min_shot_frames and nxt < dur and not f_in:
            nxt = min(dur, t + min_shot_frames / fps)
        if nxt <= t + 1e-3:
            break
        if nxt < dur - 0.05:
            cuts_t.append(nxt)
        t = nxt
    # take assignment
    usage: dict[str, int] = {}
    prev_clip, prev_size = None, None
    cuts = []
    for i, a in enumerate(cuts_t):
        b = cuts_t[i + 1] if i + 1 < len(cuts_t) else dur
        sec = _section_at(analysis, a)
        name = sec.get("name", sec["label"])
        want_broll = name in ("intro", "breakdown", "outro") or (
            name not in ("chorus",) and (i * broll_ratio) % 1 + broll_ratio >= 1)
        cands = [c for c in usable if c["song_in"] <= a + 0.02 and c["song_out"] >= b - 0.02]
        if want_broll:
            pref = [c for c in cands if c.get("role") == "broll"] or cands
        else:
            pref = [c for c in cands if c.get("role", "performance") == "performance"] or cands

        def score(c):
            s = usage.get(c["path"], 0) * 2.0
            if c["path"] == prev_clip:
                s += 10
            sz = shot_size(c["path"])
            if prev_size and sz == prev_size and sz != "?":
                s += 3  # avoid jump cuts between same-size shots
            if name == "chorus" and a in forced and sz == "WS":
                s -= 2  # land the hook wide
            return s
        chosen = min(pref, key=score) if pref else None
        is_impact = impact is not None and abs(a - impact) < 0.01
        shot = ""
        src_in = None
        if chosen:
            usage[chosen["path"]] = usage.get(chosen["path"], 0) + 1
            spd = chosen.get("playback_speed", 1.0)
            src_in = round((a - chosen["sync"]["offset_s"]) / spd, 3)
            sz = shot_size(chosen["path"])
            shot = f"{Path(chosen['path']).stem} [{sz}{', ' + shot_move(chosen['path']) if shot_move(chosen['path']) else ''}]"
            prev_clip, prev_size = chosen["path"], sz
        else:
            shot = "NO SYNCED COVERAGE - fill with B-roll / graphics"
        bar_pos = None
        if beats:
            bi = min(range(len(beats)), key=lambda j: abs(beats[j] - a))
            first = analysis["downbeats"][0] if analysis["downbeats"] else beats[0]
            fi = min(range(len(beats)), key=lambda j: abs(beats[j] - first))
            bar_pos = (bi - fi) % bpb + 1
        cuts.append({
            "index": i + 1, "start": round(a, 3), "end": round(b, 3), "time": round(a, 3),
            "frames": round((b - a) * fps), "record_tc": seconds_to_tc(a, fps), "section": name,
            "beat_in_bar": bar_pos, "clip": chosen["path"] if chosen else None,
            "source_in": src_in, "playback_speed": chosen.get("playback_speed", 1.0) if chosen else None,
            "shot": ("HERO / IMPACT - " if is_impact else "") + shot,
        })
    merged: list[dict] = []
    for c in cuts:
        prev = merged[-1] if merged else None
        if (prev and c["clip"] and prev["clip"] == c["clip"] and prev.get("playback_speed") == c.get("playback_speed")):
            if "HERO" in c["shot"] and "HERO" not in prev["shot"]:
                prev["shot"] += f" (impact at {c['record_tc']} - only this take covers it; punch-in or speed-ramp there)"
            prev["end"] = c["end"]
            prev["frames"] = round((prev["end"] - prev["start"]) * fps)
            continue
        if prev is None and c["clip"] is None and merged:
            continue
        merged.append(c)
    for n, c in enumerate(merged, 1):
        c["index"] = n
    cuts = merged
    # stats + notes
    sections_out = []
    for s in analysis["sections"]:
        sc = [c for c in cuts if s["start"] <= c["start"] < s["end"]]
        avg = sum(c["frames"] for c in sc) / len(sc) if sc else 0
        sections_out.append({"section": s.get("name", s["label"]), "start_tc": seconds_to_tc(s["start"], fps),
                             "cuts": len(sc), "avg_shot_frames": round(avg, 1),
                             "avg_shot_s": round(avg / fps, 2) if fps else None,
                             "note": SECTION_NOTES.get(s.get("name", ""), "")})
    gaps = [c for c in cuts if c["clip"] is None]
    return {
        "style": style, "style_brief": st["desc"], "pace": pace, "fps": fps,
        "total_cuts": len(cuts), "asl_s": round(dur / max(1, len(cuts)), 2),
        "sections": sections_out, "cuts": cuts,
        "coverage_gaps": [{"from": g["start"], "to": g["end"], "section": g["section"]} for g in gaps],
        "clip_usage": {Path(k).name: v for k, v in sorted(usage.items(), key=lambda kv: -kv[1])},
    }
