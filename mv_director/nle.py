"""Exports editors can import: FCPXML (Final Cut / Resolve / Premiere), CMX3600 EDL,
DaVinci Resolve marker EDL, and a CSV of markers."""
from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import quoteattr

from .media import seconds_to_tc, tc_to_seconds

# DaVinci Resolve marker colours
RESOLVE_COLORS = {
    "intro": "ResolveColorCyan", "verse": "ResolveColorBlue", "pre-chorus": "ResolveColorYellow",
    "chorus": "ResolveColorRed", "bridge": "ResolveColorPurple", "breakdown": "ResolveColorGreen",
    "outro": "ResolveColorCyan", "impact": "ResolveColorPink", "bar": "ResolveColorSand",
    "cut": "ResolveColorCream", "lyric": "ResolveColorLavender",
}


def frame_duration(fps: float) -> Fraction:
    for nominal in (24, 25, 30, 48, 50, 60):
        if abs(fps - nominal * 1000 / 1001) < 0.01:
            return Fraction(1001, nominal * 1000)
    return Fraction(1, round(fps)) if abs(fps - round(fps)) < 0.01 else Fraction(1, 1) / Fraction(fps).limit_denominator(1001)


def rt(seconds: float, fps: float) -> str:
    """Seconds -> FCPXML rational time snapped to whole frames."""
    fd = frame_duration(fps)
    frames = round(Fraction(seconds).limit_denominator(100000) / fd)
    t = frames * fd
    return "0s" if t == 0 else f"{t.numerator}/{t.denominator}s" if t.denominator != 1 else f"{t.numerator}s"


def build_markers(analysis: dict, include_bars: bool = True, include_beats: bool = False,
                  cut_plan: list | None = None, lyrics: list | None = None) -> list[dict]:
    m: list[dict] = []
    for s in analysis.get("sections", []):
        m.append({"time": s["start"], "name": f"{s.get('name', s['label']).upper()} ({s['label']})",
                  "type": s.get("name", "verse"), "note": f"{s['bars']} bars, energy {s['energy']}",
                  "duration": s["end"] - s["start"]})
    imp = analysis.get("impact_moment")
    if imp:
        m.append({"time": imp["time"], "name": "IMPACT / biggest lift", "type": "impact",
                  "note": "Hero shot or hard cut here", "duration": 0})
    if include_bars:
        for i, t in enumerate(analysis.get("bar_starts", [])):
            m.append({"time": t, "name": f"Bar {i + 1}", "type": "bar", "note": "", "duration": 0})
    if include_beats:
        for t in analysis.get("beats", []):
            m.append({"time": t, "name": "beat", "type": "bar", "note": "", "duration": 0})
    for c in cut_plan or []:
        m.append({"time": c["time"], "name": f"CUT {c['index']}", "type": "cut",
                  "note": c.get("shot", ""), "duration": 0})
    for ly in lyrics or []:
        m.append({"time": ly["start"], "name": ly["text"][:60], "type": "lyric", "note": "", "duration": 0})
    m.sort(key=lambda x: x["time"])
    return m


def write_csv(markers: list[dict], path: Path, fps: float) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timecode", "seconds", "frame", "name", "type", "note"])
        for mk in markers:
            w.writerow([seconds_to_tc(mk["time"], fps), f"{mk['time']:.3f}", round(mk["time"] * fps),
                        mk["name"], mk["type"], mk["note"]])
    return path


def write_resolve_marker_edl(markers: list[dict], path: Path, fps: float, title: str,
                             start_tc: str = "01:00:00:00") -> Path:
    """Timeline > Import > Timeline Markers from EDL (DaVinci Resolve)."""
    base = tc_to_seconds(start_tc, fps)
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for i, mk in enumerate(markers, 1):
        a = seconds_to_tc(base + mk["time"], fps)
        b = seconds_to_tc(base + mk["time"] + 1 / fps, fps)
        dur = max(1, round(mk.get("duration", 0) * fps)) if mk["type"] not in ("bar", "cut") else 1
        lines.append(f"{i:03d}  001      V     C        {a} {b} {a} {b}  ")
        name = mk["name"].replace("|", "/")
        lines.append(f" |C:{RESOLVE_COLORS.get(mk['type'], 'ResolveColorBlue')} |M:{name} |D:{dur}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def _asset_xml(aid: str, fmt_id: str | None, media: dict, fps: float) -> str:
    v, a = media.get("video"), media.get("audio")
    start = 0.0
    tc = (v or {}).get("timecode") or media.get("timecode")
    if tc and v:
        try:
            start = tc_to_seconds(tc, v["fps"])
        except Exception:
            start = 0.0
    attrs = [f'id="{aid}"', f'name={quoteattr(Path(media["file"]).stem)}', f'start="{rt(start, fps)}"',
             f'duration="{rt(media["duration_s"], fps)}"']
    if v:
        attrs += ['hasVideo="1"', f'format="{fmt_id}"']
    if a:
        attrs += ['hasAudio="1"', 'audioSources="1"', f'audioChannels="{a.get("channels") or 2}"',
                  f'audioRate="{a.get("sample_rate") or 48000}"']
    uri = Path(media["file"]).resolve().as_uri()
    return (f"    <asset {' '.join(attrs)}>\n      <media-rep kind=\"original-media\" src={quoteattr(uri)}/>\n"
            f"    </asset>\n"), start


def write_fcpxml(path: Path, title: str, fps: float, width: int, height: int, song_media: dict,
                 markers: list[dict], clips: list[dict] | None = None, cuts: list[dict] | None = None) -> Path:
    """FCPXML 1.9. Song on a connected audio lane with markers; optional synced cut on V1."""
    fd = frame_duration(fps)
    res = [f'    <format id="r1" frameDuration="{fd.numerator}/{fd.denominator}s" width="{width}" height="{height}"/>']
    song_xml, song_start = _asset_xml("a_song", None, song_media, fps)
    res.append(song_xml.rstrip("\n"))
    clip_ids: dict[str, tuple[str, float]] = {}
    for i, c in enumerate(clips or []):
        media = c["media"]
        v = media.get("video") or {}
        fid = f"f{i}"
        cfd = frame_duration(v.get("fps") or fps)
        res.append(f'    <format id="{fid}" frameDuration="{cfd.numerator}/{cfd.denominator}s" '
                   f'width="{v.get("width", width)}" height="{v.get("height", height)}"/>')
        x, st = _asset_xml(f"c{i}", fid, media, fps)
        res.append(x.rstrip("\n"))
        clip_ids[c["path"]] = (f"c{i}", st)
    total = song_media["duration_s"]
    mk_xml = []
    for mk in markers:
        tag = "chapter-marker" if mk["type"] in ("intro", "verse", "pre-chorus", "chorus", "bridge", "breakdown", "outro") else "marker"
        extra = ' posterOffset="0s"' if tag == "chapter-marker" else ""
        mk_xml.append(f'            <{tag} start="{rt(song_start + mk["time"], fps)}" duration="{rt(1 / fps, fps)}" '
                      f'value={quoteattr(mk["name"])}{extra}' + (f' note={quoteattr(mk["note"])}' if mk.get("note") else "") + "/>")
    inner = []
    inner.append(f'          <asset-clip ref="a_song" lane="-1" offset="0s" start="{rt(song_start, fps)}" '
                 f'duration="{rt(total, fps)}" name={quoteattr(Path(song_media["file"]).stem)} audioRole="music">\n'
                 + "\n".join(mk_xml) + "\n          </asset-clip>")
    for c in cuts or []:
        if c.get("clip") not in clip_ids:
            continue
        cid, st = clip_ids[c["clip"]]
        inner.append(f'          <asset-clip ref="{cid}" lane="1" offset="{rt(c["start"], fps)}" '
                     f'start="{rt(st + c["source_in"], fps)}" duration="{rt(c["end"] - c["start"], fps)}" '
                     f'name={quoteattr(Path(c["clip"]).stem)} audioRole="dialogue" enabled="1">'
                     f'<adjust-volume amount="-96dB"/></asset-clip>')
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
{chr(10).join(res)}
  </resources>
  <library>
    <event name={quoteattr(title)}>
      <project name={quoteattr(title)}>
        <sequence format="r1" duration="{rt(total, fps)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
          <gap name="Song" offset="0s" start="0s" duration="{rt(total, fps)}">
{chr(10).join(inner)}
          </gap>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
'''
    path.write_text(xml)
    return path


def write_cmx3600(path: Path, title: str, fps: float, cuts: list[dict], clip_media: dict[str, dict],
                  song_media: dict | None = None, start_tc: str = "01:00:00:00") -> Path:
    base = tc_to_seconds(start_tc, fps)
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    n = 0
    for c in cuts:
        media = clip_media.get(c.get("clip"))
        if not media:
            continue
        n += 1
        v = media.get("video") or {}
        tc = v.get("timecode") or media.get("timecode")
        src0 = tc_to_seconds(tc, v.get("fps") or fps) if tc else 0.0
        reel = "".join(ch for ch in Path(c["clip"]).stem.upper() if ch.isalnum())[:8] or "AX"
        si, so = src0 + c["source_in"], src0 + c["source_in"] + (c["end"] - c["start"])
        ri, ro = base + c["start"], base + c["end"]
        lines.append(f"{n:03d}  {reel:<8} V     C        {seconds_to_tc(si, fps)} {seconds_to_tc(so, fps)} "
                     f"{seconds_to_tc(ri, fps)} {seconds_to_tc(ro, fps)}")
        lines.append(f"* FROM CLIP NAME: {Path(c['clip']).name}")
        if c.get("shot"):
            lines.append(f"* COMMENT: {c['shot']}")
        lines.append("")
    if song_media:
        n += 1
        d = song_media["duration_s"]
        lines.append(f"{n:03d}  SONG     A     C        {seconds_to_tc(0, fps)} {seconds_to_tc(d, fps)} "
                     f"{seconds_to_tc(base, fps)} {seconds_to_tc(base + d, fps)}")
        lines.append(f"* FROM CLIP NAME: {Path(song_media['file']).name}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path
