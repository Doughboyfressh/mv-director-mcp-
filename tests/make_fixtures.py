"""Generate a synthetic 'song' with known tempo/structure and fake camera takes shot to playback."""
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 48000
BPM = 124.0
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
OUT.mkdir(parents=True, exist_ok=True)

beat = 60 / BPM
# (name, bars, energy, chord root semitones)
PLAN = [("intro", 8, 0.25, [0, 0, 5, 5]), ("verse", 8, 0.45, [0, 9, 5, 7]),
        ("chorus", 8, 1.0, [5, 7, 0, 9]), ("verse", 8, 0.45, [0, 9, 5, 7]),
        ("chorus", 8, 1.0, [5, 7, 0, 9]), ("outro", 4, 0.2, [0, 0, 0, 0])]
total_bars = sum(p[1] for p in PLAN)
lead_in = 1.0
dur = lead_in + total_bars * 4 * beat + 1.0
y = np.zeros(int(dur * SR))
rng = np.random.default_rng(1)


def add(sig, t):
    i = int(t * SR)
    j = min(len(y), i + len(sig))
    y[i:j] += sig[: j - i]


tk = np.arange(int(0.25 * SR)) / SR
kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 12)
ts = np.arange(int(0.18 * SR)) / SR
snare = rng.normal(0, 1, len(ts)) * np.exp(-ts * 22) * 0.6
th = np.arange(int(0.05 * SR)) / SR
hat = rng.normal(0, 1, len(th)) * np.exp(-th * 80) * 0.15

t = lead_in
for name, bars, e, roots in PLAN:
    for b in range(bars):
        root = roots[b % 4]
        tb = np.arange(int(4 * beat * SR)) / SR
        f0 = 220 * 2 ** (root / 12)
        pad = sum(np.sin(2 * np.pi * f0 * r * tb) for r in (1, 1.26, 1.5)) * 0.08 * (0.5 + e)
        add(pad * np.minimum(1, tb * 8), t)
        for k in range(4):
            bt = t + k * beat
            if e > 0.3 or k == 0:
                add(kick * (0.6 + 0.6 * e), bt)
            if e > 0.4 and k in (1, 3):
                add(snare * (0.5 + e), bt)
            if e > 0.9:
                add(hat, bt + beat / 2)
                add(hat, bt)
        t += 4 * beat
y = 0.8 * y / np.abs(y).max()
song = OUT / "song.wav"
pcm = (y * 32767).astype("<i2").tobytes()
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                "-ac", "2", str(song)], input=pcm, check=True)


def take(name, start_song, length, color, speed=1.0, fps=24, vfr=False):
    """Camera take: room-ish scratch audio of the song starting at start_song (may be <0)."""
    seg = np.zeros(int(length * SR))
    src_start = int(start_song * SR)
    idx = (src_start + (np.arange(len(seg)) * speed).astype(int))
    valid = (idx >= 0) & (idx < len(y))
    seg[valid] = y[idx[valid]]
    # room: lowpass-ish smear + reverb tail + noise
    ir = np.exp(-np.arange(int(0.25 * SR)) / (0.06 * SR)) * rng.normal(0, 1, int(0.25 * SR)) * 0.02
    ir[0] = 1
    seg = np.convolve(seg, ir)[: len(seg)] * 0.4 + rng.normal(0, 0.01, len(seg))
    wav = OUT / f"{name}.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    str(wav)], input=(np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes(), check=True)
    vf = f"color=c={color}:s=640x360:r={fps}:d={length},drawtext=text='{name} %{{pts\\:hms}}':fontsize=28:fontcolor=white:x=20:y=20"
    args = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", vf, "-i", str(wav),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    args += [str(OUT / f"{name}.mp4")]
    subprocess.run(args, check=True)
    wav.unlink()


take("A001_wide", -2.5, 40, "darkblue")      # rolled before playback started
take("A002_close", 20.0, 30, "darkred")
take("A003_handheld", 45.0, 35, "darkgreen")
take("B001_slowmo_2x", 30.0, 10, "purple", speed=2.0, fps=48)
print("song", song, "true bpm", BPM, "first downbeat", lead_in)
print("expected offsets: A001 -2.5, A002 20.0, A003 45.0, B001 30.0 @2x")
