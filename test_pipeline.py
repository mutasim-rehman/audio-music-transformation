"""
Quick end-to-end test for the unified pipeline.

Generates synthetic signals, runs transform_audio, and checks basic
structural invariants.  No real audio files required.

Run:
    python test_pipeline.py
"""

import sys
import os
import tempfile
from pathlib import Path

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def generate_test_wavs(tmp_dir: str, sr: int = 22050, duration: float = 3.0):
    """Create two tiny WAV files: white-noise base & sine-melody projected."""
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # base: low-pass-filtered noise (cloud-like)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(t)) * 0.3
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 1000 / (sr / 2), btype="low")
    base = filtfilt(b, a, noise).astype(np.float32)

    # projected: simple two-note melody
    melody = np.zeros_like(t, dtype=np.float32)
    half = len(t) // 2
    melody[:half] = 0.5 * np.sin(2 * np.pi * 330 * t[:half])   # E4
    melody[half:] = 0.5 * np.sin(2 * np.pi * 440 * t[half:])   # A4
    # apply simple envelope per note
    env = np.exp(-2.0 * np.mod(t, duration / 2))
    melody *= env.astype(np.float32)

    base_path = os.path.join(tmp_dir, "test_base.wav")
    proj_path = os.path.join(tmp_dir, "test_projected.wav")
    sf.write(base_path, base, sr)
    sf.write(proj_path, melody, sr)
    return base_path, proj_path, sr


def test_pipeline():
    from src.pipeline import transform_audio

    print("=" * 50)
    print("  Pipeline Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmp:
        base_path, proj_path, sr = generate_test_wavs(tmp)
        out_path = os.path.join(tmp, "test_output.wav")

        y_out, info = transform_audio(
            base_path=base_path,
            projected_path=proj_path,
            output_path=out_path,
            alpha=0.7,
            preset="balanced",
            sr=sr,
        )

        # ── assertions ────────────────────────────────────────────────
        errors = []

        # 1. output exists
        if not os.path.isfile(out_path):
            errors.append("Output file was not created")

        # 2. length matches projected
        import soundfile as sf
        proj_data, _ = sf.read(proj_path)
        expected_len = len(proj_data)
        if len(y_out) != expected_len:
            errors.append(
                f"Length mismatch: output={len(y_out)}, expected={expected_len}"
            )

        # 3. not silent
        rms = np.sqrt(np.mean(y_out ** 2))
        if rms < 1e-6:
            errors.append(f"Output is essentially silent (RMS={rms:.2e})")

        # 4. no NaN / Inf
        if not np.all(np.isfinite(y_out)):
            errors.append("Output contains NaN or Inf values")

        # 5. within [-1, 1]
        if np.max(np.abs(y_out)) > 1.0:
            errors.append(f"Output clipped: max={np.max(np.abs(y_out)):.4f}")

        if errors:
            print("\n❌ FAIL")
            for e in errors:
                print(f"   • {e}")
            return False
        else:
            print(f"\n✅ ALL CHECKS PASSED")
            print(f"   output length : {len(y_out)} samples")
            print(f"   output RMS    : {rms:.4f}")
            print(f"   output peak   : {np.max(np.abs(y_out)):.4f}")
            return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
