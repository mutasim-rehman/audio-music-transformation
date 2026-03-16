"""
Unified Audio Style Projection Pipeline

Combines the best techniques from all three existing approaches into
one high-quality transformation function:

  Layer 1 (Skeleton) — Additive harmonic synthesis at target pitches,
                        coloured by the source's spectral shape.
  Layer 2 (Flesh)    — Spectral reshaping of the actual source audio
                        toward the harmonic structure of the target.
  Layer 3 (Breath)   — Envelope + formant transfer for natural phrasing.
  Layer 4 (Mix)      — Alpha-controlled blend with crossfade.

Usage:
    from src.pipeline import transform_audio
    y_out, info = transform_audio("samples/source_clouds.wav",
                                  "samples/target_elvis.wav",
                                  "outputs/result.wav",
                                  alpha=0.7)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirpeak, lfilter, medfilt
from scipy.ndimage import uniform_filter1d

# ─── local imports ────────────────────────────────────────────────────
from .audio_io import load_audio, save_audio
from .analysis import compute_stft, extract_pitch, extract_envelope

import librosa


# ======================================================================
#  PRESETS
# ======================================================================

PRESETS = {
    "balanced": {
        "harmonic_weight": 0.55,    # weight of synthesised harmonics layer
        "reshape_weight": 0.35,     # weight of spectral-reshaped source layer
        "envelope_strength": 0.65,  # how much target dynamics are imposed
        "formant_strength": 0.25,   # vocal formant resonance intensity
        "n_harmonics": 12,
        "harmonic_rolloff": 0.7,
    },
    "more-source": {
        "harmonic_weight": 0.30,
        "reshape_weight": 0.20,
        "envelope_strength": 0.40,
        "formant_strength": 0.15,
        "n_harmonics": 8,
        "harmonic_rolloff": 0.8,
    },
    "more-singing": {
        "harmonic_weight": 0.75,
        "reshape_weight": 0.45,
        "envelope_strength": 0.80,
        "formant_strength": 0.40,
        "n_harmonics": 15,
        "harmonic_rolloff": 0.6,
    },
}


# ======================================================================
#  PUBLIC API
# ======================================================================

def transform_audio(
    base_path: Union[str, Path],
    projected_path: Union[str, Path],
    output_path: Union[str, Path] = "outputs/output.wav",
    alpha: float = 0.7,
    preset: str = "balanced",
    sr: int = 22050,
    hop_length: int = 256,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, dict]:
    """
    Transform *base* audio so that it "sounds like" the *projected* audio
    while keeping the intrinsic texture of the base.

    Parameters
    ----------
    base_path : path-like
        Source / environmental audio (clouds, wind, rain …).
    projected_path : path-like
        Target / vocal audio whose melody & rhythm the output will follow.
    output_path : path-like
        Where to write the resulting WAV file.
    alpha : float  [0 … 1]
        Overall transformation intensity.
        0 → pure base, 1 → maximum singing effect.
    preset : str
        One of ``"balanced"``, ``"more-source"``, ``"more-singing"``.
    sr : int
        Sample rate for processing.
    hop_length : int
        STFT hop size (smaller → better time resolution).
    n_fft : int
        FFT window size.

    Returns
    -------
    y_output : np.ndarray
        The transformed audio (same length as *projected*).
    info : dict
        Intermediate data useful for visualisation / debugging.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
        )
    cfg = PRESETS[preset]

    # ── load audio ────────────────────────────────────────────────────
    print("[PIPELINE] Loading audio …")
    y_base, _ = load_audio(base_path, sr=sr)
    y_proj, _ = load_audio(projected_path, sr=sr)

    n_target = len(y_proj)
    duration = n_target / sr
    print(f"[PIPELINE] Target duration: {int(duration // 60)}:{duration % 60:05.2f}")

    # Loop base to match projected length
    if len(y_base) < n_target:
        repeats = int(np.ceil(n_target / len(y_base)))
        y_base = np.tile(y_base, repeats)[:n_target]
        print(f"[PIPELINE] Base looped {repeats}× to match projected length")
    else:
        y_base = y_base[:n_target]

    info: dict = {}

    # ── analyse projected audio ───────────────────────────────────────
    print("[PIPELINE] Analysing projected audio …")

    f0, voiced, times_f0 = extract_pitch(
        y_proj, sr, hop_length=hop_length, fmin=50, fmax=800
    )
    info["f0"] = f0
    info["voiced"] = voiced

    env, times_env = extract_envelope(y_proj, sr, hop_length=hop_length)
    info["envelope"] = env

    # smooth pitch for natural transitions
    f0_smooth = _smooth_pitch(f0, voiced, smoothing=0.8)

    # ── analyse base texture ──────────────────────────────────────────
    print("[PIPELINE] Analysing base texture …")

    S_base = librosa.stft(y_base, n_fft=n_fft, hop_length=hop_length)
    mag_base = np.abs(S_base)
    phase_base = np.angle(S_base)

    spectral_shape = np.mean(mag_base, axis=1)
    spectral_shape /= np.max(spectral_shape) + 1e-10
    info["spectral_shape"] = spectral_shape

    # ── LAYER 1: harmonic synthesis ───────────────────────────────────
    print("[PIPELINE] Layer 1 — Harmonic synthesis …")
    y_harmonics = _synthesise_harmonics(
        n_target, sr, f0_smooth, voiced, times_f0,
        spectral_shape, n_fft,
        n_harmonics=cfg["n_harmonics"],
        rolloff=cfg["harmonic_rolloff"],
        hop_length=hop_length,
    )
    info["harmonics"] = y_harmonics

    # ── LAYER 2: spectral reshaping of source ─────────────────────────
    print("[PIPELINE] Layer 2 — Spectral reshaping …")
    n_frames = min(mag_base.shape[1], len(f0_smooth))
    mag_reshaped = _reshape_spectrum(
        mag_base[:, :n_frames],
        f0_smooth, voiced, n_frames,
        sr, n_fft,
        strength=alpha * cfg["reshape_weight"],
    )
    S_reshaped = mag_reshaped * np.exp(1j * phase_base[:, :n_frames])
    y_reshaped = librosa.istft(S_reshaped, hop_length=hop_length, length=n_target)
    info["reshaped"] = y_reshaped

    # ── LAYER 3: envelope & formant shaping ───────────────────────────
    print("[PIPELINE] Layer 3 — Envelope & formant transfer …")

    # interpolate envelope to sample-rate
    t = np.arange(n_target) / sr
    env_interp = interp1d(times_env, env, kind="linear",
                          fill_value=0, bounds_error=False)
    env_at_samples = env_interp(t)
    env_at_samples /= np.max(env_at_samples) + 1e-10

    env_smooth = _smooth_envelope(env_at_samples, sr)

    # shape harmonics and reshaped source with envelope
    y_harmonics_shaped = y_harmonics * env_smooth
    y_reshaped_shaped = y_reshaped * (
        cfg["envelope_strength"] * env_smooth
        + (1.0 - cfg["envelope_strength"])
    )

    # apply formant resonances to harmonics
    if cfg["formant_strength"] > 0:
        y_harmonics_shaped = _apply_formants(
            y_harmonics_shaped, sr, strength=cfg["formant_strength"]
        )

    # ── LAYER 4: mix ──────────────────────────────────────────────────
    print("[PIPELINE] Layer 4 — Mixing …")

    hw = alpha * cfg["harmonic_weight"]
    rw = alpha * cfg["reshape_weight"]
    sw = 1.0 - alpha  # source weight

    y_mix = (
        hw * y_harmonics_shaped
        + rw * y_reshaped_shaped
        + sw * y_base[:n_target]
    )

    # normalise
    peak = np.max(np.abs(y_mix)) + 1e-10
    y_output = y_mix / peak * 0.9

    # ensure exact length
    if len(y_output) < n_target:
        y_output = np.pad(y_output, (0, n_target - len(y_output)))
    else:
        y_output = y_output[:n_target]

    info["y_output"] = y_output
    info["duration_seconds"] = duration

    # ── save ──────────────────────────────────────────────────────────
    save_audio(output_path, y_output, sr=sr)
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"[PIPELINE] ✓ Done — {minutes}:{seconds:05.2f} saved to {output_path}")

    return y_output, info


# ======================================================================
#  INTERNAL HELPERS
# ======================================================================

def _smooth_pitch(
    f0: np.ndarray, voiced: np.ndarray, smoothing: float = 0.8
) -> np.ndarray:
    """Interpolate through unvoiced gaps and lightly smooth the contour."""
    f0s = f0.copy()
    vi = np.where(voiced)[0]
    if len(vi) > 1:
        fn = interp1d(vi, f0[vi], kind="linear",
                      fill_value=0, bounds_error=False)
        uvi = np.where(~voiced)[0]
        for idx in uvi:
            if vi[0] < idx < vi[-1]:
                f0s[idx] = fn(idx)
    if smoothing > 0:
        win = max(1, int(len(f0s) * smoothing * 0.05))
        f0s = uniform_filter1d(f0s, size=win)
    return f0s


def _smooth_envelope(
    envelope: np.ndarray, sr: int,
    attack_ms: float = 10.0, release_ms: float = 50.0,
) -> np.ndarray:
    """One-pole attack/release smoother."""
    attack_coef = 1.0 - np.exp(-1.0 / max(1, int(sr * attack_ms / 1000)))
    release_coef = 1.0 - np.exp(-1.0 / max(1, int(sr * release_ms / 1000)))
    out = np.empty_like(envelope)
    cur = 0.0
    for i, val in enumerate(envelope):
        coef = attack_coef if val > cur else release_coef
        cur += coef * (val - cur)
        out[i] = cur
    return out


def _synthesise_harmonics(
    n_samples: int,
    sr: int,
    f0: np.ndarray,
    voiced: np.ndarray,
    times_f0: np.ndarray,
    spectral_shape: np.ndarray,
    n_fft: int,
    n_harmonics: int = 12,
    rolloff: float = 0.7,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Additive synthesis at target pitches, coloured by source spectral shape.
    Uses phase accumulators for click-free output.
    """
    t = np.arange(n_samples) / sr

    # interpolate pitch to sample rate
    if len(f0) > 1:
        f0_interp = interp1d(times_f0, f0, kind="linear",
                             fill_value=0, bounds_error=False)(t)
    else:
        f0_interp = np.zeros(n_samples)

    voiced_interp = interp1d(
        times_f0, voiced.astype(float),
        kind="nearest", fill_value=0, bounds_error=False
    )(t) > 0.5

    freq_bins = np.linspace(0, sr / 2, len(spectral_shape))

    # harmonic weights (natural voice roll-off)
    hw = np.array([1.0 / ((h + 1) ** rolloff) for h in range(n_harmonics)])
    hw /= hw.sum()

    y = np.zeros(n_samples)
    phases = np.zeros(n_harmonics)
    block_size = hop_length

    for blk in range(n_samples // block_size):
        s = blk * block_size
        e = min(s + block_size, n_samples)
        f0_blk = np.mean(f0_interp[s:e])
        is_voiced = np.mean(voiced_interp[s:e]) > 0.5

        if f0_blk > 20 and is_voiced:
            for h in range(n_harmonics):
                hf = f0_blk * (h + 1)
                if hf >= sr / 2:
                    break
                # texture colouring
                idx = min(np.argmin(np.abs(freq_bins - hf)),
                          len(spectral_shape) - 1)
                tw = 0.3 + 0.7 * spectral_shape[idx]

                inc = 2 * np.pi * hf / sr
                ph = phases[h] + np.cumsum(np.full(e - s, inc))
                phases[h] = ph[-1] % (2 * np.pi)

                y[s:e] += hw[h] * tw * np.sin(ph)

    mx = np.max(np.abs(y))
    if mx > 0:
        y /= mx
    return y


def _reshape_spectrum(
    mag: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
    n_frames: int,
    sr: int,
    n_fft: int,
    strength: float = 0.3,
    n_harmonics: int = 12,
) -> np.ndarray:
    """
    Pull existing source energy toward harmonic frequencies of the target
    pitch — a 'spectral attractor' reshaping.
    """
    nf = mag.shape[0]
    freqs = np.linspace(0, sr / 2, nf)
    out = mag.copy()

    # resample pitch arrays to n_frames
    if len(f0) != n_frames:
        f0 = np.interp(np.linspace(0, 1, n_frames),
                        np.linspace(0, 1, len(f0)), f0)
    if len(voiced) != n_frames:
        voiced = np.interp(np.linspace(0, 1, n_frames),
                           np.linspace(0, 1, len(voiced)),
                           voiced.astype(float)) > 0.5

    for t in range(n_frames):
        if f0[t] < 50 or not voiced[t]:
            out[:, t] *= (1.0 - strength * 0.5)  # quiet unvoiced
            continue

        attractor = np.zeros(nf)
        for h in range(1, n_harmonics + 1):
            hf = f0[t] * h
            if hf >= sr / 2:
                break
            bi = np.argmin(np.abs(freqs - hf))
            bw = max(3, int(nf * 0.015 / h))
            for off in range(-bw, bw + 1):
                idx = bi + off
                if 0 <= idx < nf:
                    w = np.exp(-0.5 * (off / (bw / 2 + 1e-6)) ** 2)
                    s = 1.0 / (h ** 0.5)
                    attractor[idx] = max(attractor[idx], w * s)

        mx = np.max(attractor)
        if mx > 0:
            attractor /= mx

        modulation = (1.0 - strength) + strength * attractor
        out[:, t] *= modulation

    return out


def _apply_formants(
    y: np.ndarray, sr: int, strength: float = 0.3
) -> np.ndarray:
    """Apply a set of typical vocal-formant resonances."""
    formants = np.array([500, 1500, 2500, 3500])
    bandwidths = np.array([100, 120, 150, 200])
    gains = np.array([1.0, 0.7, 0.5, 0.3]) * strength

    y_out = np.zeros_like(y)
    for fc, bw, g in zip(formants, bandwidths, gains):
        if fc < sr / 2:
            Q = fc / bw
            b, a = iirpeak(fc / (sr / 2), Q)
            y_out += g * lfilter(b, a, y)

    mx = np.max(np.abs(y_out))
    if mx > 0:
        y_out /= mx
    return y_out
