"""Music analysis with numpy/scipy only: tempo, beat grid, downbeats, sections, key,
energy, and playback-sync by cross-correlation."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from .media import decode_audio

SR = 22050
HOP = 512
NFFT = 2048
FPS_ENV = SR / HOP  # ~43 frames/s
# frame index -> seconds: window is start-anchored, flux peaks ~1.5 hops before the window centre
LATENCY = 0.0725  # calibrated on synthetic transients (NFFT/2 window centre + flux lead)

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------- features
def _stft_mag(y: np.ndarray, sr: int = SR, hop: int = HOP, nfft: int = NFFT):
    _, _, Z = signal.stft(y, fs=sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=True)
    return np.abs(Z).astype(np.float32)


def _mel_fb(sr: int, nfft: int, n_mels: int = 64, fmin: float = 30, fmax: float | None = None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(nfft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo + 1e-9), (hi - bins) / (hi - c + 1e-9)), 0, None)
    return fb


def _chroma(S: np.ndarray, sr: int = SR, nfft: int = NFFT) -> np.ndarray:
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    valid = (freqs > 55) & (freqs < 4200)
    pcs = (np.round(12 * np.log2(freqs[valid] / 440.0)) + 9) % 12
    C = np.zeros((12, S.shape[1]), dtype=np.float32)
    Sv = S[valid] ** 2
    for pc in range(12):
        C[pc] = Sv[pcs == pc].sum(axis=0)
    return C / (C.max(axis=0, keepdims=True) + 1e-9)


def onset_envelope(S_mel_db: np.ndarray) -> np.ndarray:
    flux = np.maximum(0, np.diff(S_mel_db, axis=1)).mean(axis=0)
    flux = np.concatenate([[0], flux])
    flux = flux - uniform_filter1d(flux, size=int(FPS_ENV))  # remove slow trend
    flux = np.maximum(flux, 0)
    return flux / (flux.std() + 1e-9)


# ---------------------------------------------------------------- tempo & beats
def estimate_tempo(env: np.ndarray, fps: float = FPS_ENV, lo: float = 60, hi: float = 200,
                   prior_bpm: float = 115) -> tuple[float, list[float]]:
    n = len(env)
    ac = np.correlate(env - env.mean(), env - env.mean(), mode="full")[n - 1:]
    ac /= ac[0] + 1e-9
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    mask = (bpm >= lo) & (bpm <= hi)
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / prior_bpm) / 0.9) ** 2)
    score = np.where(mask, ac * weight, -np.inf)
    # harmonic reinforcement: lag and 2*lag
    score2 = score.copy()
    for L in np.where(mask)[0]:
        if 2 * L < len(ac):
            score2[L] += 0.5 * ac[2 * L]
    best = int(np.argmax(score2))
    # parabolic refinement
    if 1 <= best < len(score2) - 1 and np.isfinite(score2[best - 1]) and np.isfinite(score2[best + 1]):
        a, b, c = score2[best - 1], score2[best], score2[best + 1]
        best_f = best + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        best_f = best
    tempo = 60 * fps / best_f
    cands = sorted({round(tempo / 2, 1), round(tempo, 1), round(tempo * 2, 1)} & set(
        x for x in (round(tempo / 2, 1), round(tempo, 1), round(tempo * 2, 1)) if lo / 1.5 <= x <= hi * 1.5))
    return float(tempo), cands


def track_beats(env: np.ndarray, tempo: float, fps: float = FPS_ENV, tightness: float = 100) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns beat frame indices."""
    period = 60 * fps / tempo
    n = len(env)
    local = np.convolve(env, np.exp(-0.5 * ((np.arange(-period, period + 1)) / (period / 32)) ** 2), "same")
    score = local.copy()
    back = -np.ones(n, dtype=int)
    win = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    txcost = -tightness * np.log(-win / period) ** 2
    for i in range(n):
        idx = i + win
        ok = idx >= 0
        if not ok.any():
            continue
        cand = score[idx[ok]] + txcost[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[i] = local[i] + cand[j]
            back[i] = idx[ok][j]
    # start from the best of the last period
    tail = np.arange(max(0, n - int(period)), n)
    i = int(tail[np.argmax(score[tail])])
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    # trim weak leading/trailing beats (silence)
    thr = 0.3 * np.median(local[beats]) if len(beats) else 0
    keep = np.where(local[beats] > thr)[0]
    if len(keep):
        beats = beats[keep[0]: keep[-1] + 1]
    return beats


def find_downbeats(beats: np.ndarray, low_env: np.ndarray, chroma: np.ndarray, beats_per_bar: int = 4) -> int:
    """Return phase (0..beats_per_bar-1) of the first downbeat in the beat list."""
    if len(beats) < beats_per_bar * 2:
        return 0
    edges = list(beats) + [beats[-1] + int(np.median(np.diff(beats)))]
    lo = np.array([low_env[a:a + 3].max() if a < len(low_env) else 0 for a in beats])
    ch = np.stack([chroma[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    ch = ch / (np.linalg.norm(ch, axis=0, keepdims=True) + 1e-9)
    chg = np.concatenate([[0], 1 - (ch[:, 1:] * ch[:, :-1]).sum(axis=0)])
    feat = 0.5 * lo / (lo.std() + 1e-9) + chg / (chg.std() + 1e-9)
    scores = [feat[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    return int(np.argmax(scores))


def _refine_peaks(env: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = idx.astype(float)
    for k, i in enumerate(idx):
        if 1 <= i < len(env) - 1:
            a, b, c = env[i - 1], env[i], env[i + 1]
            d = a - 2 * b + c
            if d < 0:
                out[k] = i + float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    return out


# ---------------------------------------------------------------- key
def estimate_key(chroma: np.ndarray) -> dict:
    prof = chroma.mean(axis=1)
    best = (-2.0, "", "")
    for i in range(12):
        for name, tmpl in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(prof, np.roll(tmpl, i))[0, 1]
            if r > best[0]:
                best = (r, PITCHES[i], name)
    return {"key": f"{best[1]} {best[2]}", "confidence": round(float(best[0]), 2)}


# ---------------------------------------------------------------- sections
def _checkerboard(L: int) -> np.ndarray:
    g = signal.windows.gaussian(2 * L, std=L / 2)
    G = np.outer(g, g)
    sign = np.ones((2 * L, 2 * L))
    sign[:L, L:] = sign[L:, :L] = -1
    return G * sign


def segment_sections(bar_feats: np.ndarray, bar_energy: np.ndarray, min_bars: int = 4) -> tuple[list[int], list[str]]:
    """Novelty-based boundaries on bar-synchronous features, then cluster labels."""
    nb = bar_feats.shape[1]
    if nb < 2 * min_bars:
        return [0], ["A"]
    Z = (bar_feats - bar_feats.mean(axis=1, keepdims=True)) / (bar_feats.std(axis=1, keepdims=True) + 1e-9)
    X = Z / (np.linalg.norm(Z, axis=0, keepdims=True) + 1e-9)
    SSM = X.T @ X
    L = min(4, nb // 4) or 1
    K = _checkerboard(L)
    pad = np.pad(SSM, L, mode="edge")
    nov = np.array([(pad[i:i + 2 * L, i:i + 2 * L] * K).sum() for i in range(nb)])
    e = np.abs(np.diff(np.concatenate([[bar_energy[0]], bar_energy])))
    nov = nov / (nov.max() + 1e-9) + 0.5 * e / (e.max() + 1e-9)
    # prefer boundaries on 4-bar phrase grid
    nov = nov * np.array([1.25 if i % 4 == 0 else (1.0 if i % 2 == 0 else 0.7) for i in range(nb)])
    peaks, _ = signal.find_peaks(nov, distance=min_bars, height=np.percentile(nov, 60))
    bounds = [0] + [int(p) for p in peaks if min_bars <= p <= nb - min_bars]
    # merge too-short
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= min_bars:
            clean.append(b)
    bounds = clean
    # section descriptors -> greedy clustering
    segs = list(zip(bounds, bounds[1:] + [nb]))
    # sequence-aware similarity: compare bar k of one section with bar k of another
    seg_e = [float(bar_energy[a:b].mean()) for a, b in segs]

    def sim(i: int, j: int) -> float:
        (a1, b1), (a2, b2) = segs[i], segs[j]
        n = min(b1 - a1, b2 - a2)
        diag = float(np.mean([SSM[a1 + k, a2 + k] for k in range(n)]))
        return diag - 1.5 * abs(seg_e[i] - seg_e[j]) - 0.05 * abs((b1 - a1) - (b2 - a2))

    labels: list[str] = []
    protos: list[int] = []
    for i in range(len(segs)):
        sims = [sim(i, p) for p in protos]
        if sims and max(sims) > 0.55:
            labels.append(chr(65 + int(np.argmax(sims))))
        else:
            protos.append(i)
            labels.append(chr(65 + len(protos) - 1))
    return bounds, labels


def name_sections(segs: list[dict]) -> None:
    """Heuristic musical names (intro/verse/chorus/...). Always marked as a guess."""
    if not segs:
        return
    energies = np.array([s["energy"] for s in segs])
    counts = {}
    for s in segs:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    label_energy = {lab: np.mean([s["energy"] for s in segs if s["label"] == lab]) for lab in counts}
    repeated = [lab for lab in counts if counts[lab] > 1]
    chorus = max(repeated, key=lambda l: label_energy[l]) if repeated else max(label_energy, key=label_energy.get)
    verse_cands = [l for l in repeated if l != chorus]
    verse = max(verse_cands, key=lambda l: counts[l]) if verse_cands else None
    hi = np.percentile(energies, 66)
    for i, s in enumerate(segs):
        if s["label"] == chorus:
            n = "chorus"
        elif s["label"] == verse:
            n = "verse"
        elif i == 0 and s["energy"] < hi:
            n = "intro"
        elif i == len(segs) - 1 and s["energy"] < hi:
            n = "outro"
        elif i + 1 < len(segs) and segs[i + 1]["label"] == chorus and s["energy"] < label_energy[chorus]:
            n = "pre-chorus"
        elif s["energy"] < np.percentile(energies, 33):
            n = "breakdown"
        else:
            n = "bridge"
        s["name"] = n


# ---------------------------------------------------------------- main entry
def analyze(path: str, beats_per_bar: int = 4, bpm_hint: float | None = None) -> dict:
    y = decode_audio(path, sr=SR)
    duration = len(y) / SR
    S = _stft_mag(y)
    fb = _mel_fb(SR, NFFT, 64)
    M = fb @ S
    Mdb = 10 * np.log10(M + 1e-10)
    env = onset_envelope(Mdb)
    low_env = np.maximum(0, np.diff(Mdb[:5], axis=1)).mean(axis=0)
    low_env = np.concatenate([[0], low_env])
    chroma = _chroma(S)
    rms = np.sqrt((S ** 2).mean(axis=0))

    tempo, cands = estimate_tempo(env, prior_bpm=bpm_hint or 115)
    if bpm_hint:
        tempo = min([tempo / 2, tempo, tempo * 2], key=lambda t: abs(t - bpm_hint))
    beats = track_beats(env, tempo)
    beat_t = beats / FPS_ENV
    beat_t = _refine_peaks(env, beats) / FPS_ENV + LATENCY
    if len(beat_t) > 8:
        slope = np.polyfit(np.arange(len(beat_t)), beat_t, 1)[0]
        tempo = float(60 / slope)
    phase = find_downbeats(beats, low_env, chroma, beats_per_bar)
    down_idx = list(range(phase, len(beats), beats_per_bar))
    downbeats = beat_t[down_idx]
    # include pickup bar start if phase>0
    bar_starts = list(downbeats)
    if phase > 0 and downbeats.size:
        bar_starts = [max(0.0, downbeats[0] - beats_per_bar * 60 / tempo)] + bar_starts

    # bar-synchronous features
    bar_frames = [max(0, int((t - LATENCY) * FPS_ENV)) for t in bar_starts] + [S.shape[1]]
    feats, energy = [], []
    for a, b in zip(bar_frames[:-1], bar_frames[1:]):
        b = max(b, a + 1)
        feats.append(np.concatenate([chroma[:, a:b].mean(axis=1), (Mdb[::4, a:b].mean(axis=1) + 100) / 100]))
        energy.append(float(np.sqrt((rms[a:b] ** 2).mean())))
    bar_feats = np.array(feats).T if feats else np.zeros((1, 1))
    energy = np.array(energy) if energy else np.zeros(1)
    e_db = 20 * np.log10(energy + 1e-9)
    e_norm = (e_db - e_db.min()) / (np.ptp(e_db) + 1e-9)

    bounds, labels = segment_sections(bar_feats, e_norm)
    sections = []
    for i, (b, lab) in enumerate(zip(bounds, labels)):
        end_bar = bounds[i + 1] if i + 1 < len(bounds) else len(bar_starts)
        start_t = 0.0 if i == 0 else bar_starts[b]
        end_t = bar_starts[end_bar] if end_bar < len(bar_starts) else duration
        sections.append({
            "label": lab, "start": round(float(start_t), 3), "end": round(float(end_t), 3),
            "bars": int(end_bar - b), "start_bar": int(b + 1),
            "energy": round(float(e_norm[b:end_bar].mean()), 2),
        })
    name_sections(sections)

    # biggest energy lift = the "drop"/impact moment (prefer section boundaries)
    impact = None
    if len(sections) > 1:
        lifts = [(sections[i]["energy"] - sections[i - 1]["energy"], i) for i in range(1, len(sections))]
        lift, i = max(lifts)
        if lift > 0:
            impact = {"time": sections[i]["start"], "bar": sections[i]["start_bar"], "lift": round(float(lift), 2),
                      "into": sections[i].get("name")}
    if impact is None and len(e_norm) > 2:
        d = np.diff(e_norm)
        i = int(np.argmax(d)) + 1
        impact = {"time": round(float(bar_starts[i]), 3), "bar": i + 1, "lift": round(float(d[i - 1]), 2)}

    lufs_env = 20 * np.log10(median_filter(rms, size=int(FPS_ENV)) + 1e-9)
    return {
        "source": str(path),
        "duration": round(duration, 3),
        "tempo_bpm": round(tempo, 2),
        "tempo_alternatives": cands,
        "beats_per_bar": beats_per_bar,
        "beat_interval_s": round(60 / tempo, 4),
        "bar_length_s": round(beats_per_bar * 60 / tempo, 4),
        "key": estimate_key(chroma),
        "first_downbeat": round(float(downbeats[0]), 3) if downbeats.size else None,
        "beats": [round(float(t), 3) for t in beat_t],
        "downbeats": [round(float(t), 3) for t in downbeats],
        "bar_starts": [round(float(t), 3) for t in bar_starts],
        "bar_energy": [round(float(x), 3) for x in e_norm],
        "sections": sections,
        "impact_moment": impact,
        "dynamic_range_db": round(float(np.percentile(lufs_env, 95) - np.percentile(lufs_env, 10)), 1),
        "notes": "Section names are heuristic guesses from repetition + energy; confirm by ear.",
    }


# ---------------------------------------------------------------- sync
SYNC_SR = 8000
SYNC_HOP = 32  # 4 ms resolution


def _sync_feats(y: np.ndarray) -> np.ndarray:
    """Per-band onset + per-band spectral shape (captures rhythm *and* harmony)."""
    S = _stft_mag(y, sr=SYNC_SR, hop=SYNC_HOP, nfft=512)
    fb = _mel_fb(SYNC_SR, 512, 24, fmin=60, fmax=3800)
    Mdb = 10 * np.log10(fb @ S + 1e-10)
    on = np.maximum(0, np.diff(Mdb, axis=1, prepend=Mdb[:, :1]))
    on = on - uniform_filter1d(on, 64, axis=1)
    shape = Mdb - Mdb.mean(axis=0, keepdims=True)          # timbre/harmony, level-independent
    shape = uniform_filter1d(shape, 8, axis=1)
    shape = shape - uniform_filter1d(shape, 1000, axis=1)   # remove room EQ / slow drift
    F = np.vstack([on, 0.5 * shape])
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-9)
    return F.astype(np.float32)


def sync_offset(master_path: str, clip_path: str, playback_speed: float = 1.0,
                max_clip_seconds: float = 120) -> dict:
    """Where in the song does this clip start? Positive offset = clip frame 0 is at song time `offset`.

    playback_speed: if the artist performed to a sped-up track (e.g. 2.0 for 2x playback shot at
    48/50/60 fps for slow motion), give that factor. The offset returned is in *song* time, and the
    clip's audio is compared against the master sped up by that factor."""
    song = decode_audio(master_path, sr=SYNC_SR)
    clip = decode_audio(clip_path, sr=SYNC_SR, dur=max_clip_seconds)
    if playback_speed != 1.0:
        n = int(len(song) / playback_speed)
        song = signal.resample(song, n).astype(np.float32)
    Fs, Fc = _sync_feats(song), _sync_feats(clip)
    ls, lc = Fs.shape[1], Fc.shape[1]
    if lc < 50:
        return {"ok": False, "reason": "clip audio too short"}
    nfft = 1 << (ls + lc - 1).bit_length()
    acc = np.zeros(nfft)
    for b in range(Fs.shape[0]):
        acc += np.fft.irfft(np.fft.rfft(Fs[b], nfft) * np.conj(np.fft.rfft(Fc[b], nfft)), nfft)
    xc = np.concatenate([acc[-(lc - 1):], acc[:ls]])
    lags = np.arange(-(lc - 1), ls)
    overlap = np.minimum(ls, lags + lc) - np.maximum(0, lags)
    xc = xc / np.maximum(overlap, lc * 0.5) / Fs.shape[0]
    i = int(np.argmax(xc))
    peak = xc[i]
    guard = int(0.2 * SYNC_SR / SYNC_HOP)
    masked = xc.copy()
    masked[max(0, i - guard): i + guard] = -np.inf
    alts = []
    for _ in range(3):
        j = int(np.argmax(masked))
        if not np.isfinite(masked[j]):
            break
        alts.append((j, float(masked[j])))
        masked[max(0, j - guard): j + guard] = -np.inf
    second = alts[0][1] if alts else 0.0
    ratio = float(peak / second) if second > 0 else 99.0
    lag_s = lags[i] * SYNC_HOP / SYNC_SR
    offset_song = lag_s * playback_speed
    conf = "high" if ratio > 1.3 else ("medium" if ratio > 1.12 else "low")
    return {
        "ok": True,
        "offset_s": round(float(offset_song), 3),
        "confidence": conf,
        "peak_ratio": round(ratio, 2),
        "playback_speed": playback_speed,
        "alternatives_s": [round(float(lags[j] * SYNC_HOP / SYNC_SR * playback_speed), 3) for j, _ in alts],
        "meaning": ("Clip frame 0 lines up with song time offset_s. "
                    "Negative = camera started rolling before the song started."),
    }
