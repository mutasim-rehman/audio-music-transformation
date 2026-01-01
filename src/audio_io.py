"""
Audio I/O Module

Handles loading, saving, and playback of audio files.

LEARNING NOTES:
- Audio is stored as a sequence of amplitude values (samples)
- Sample rate (sr) = how many samples per second (e.g., 44100 Hz)
- Higher sample rates capture more high-frequency detail
- We typically work with mono (single channel) audio for processing
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings


def load_audio(
    filepath: Union[str, Path],
    sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy array.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the audio file (supports wav, mp3, flac, etc.)
    sr : int, default=22050
        Target sample rate. Audio will be resampled if necessary.
        22050 Hz is a good balance between quality and processing speed.
    mono : bool, default=True
        If True, convert stereo to mono by averaging channels.
    duration : float, optional
        Only load this many seconds of audio.
    offset : float, default=0.0
        Start reading at this time (in seconds).
    
    Returns
    -------
    y : np.ndarray
        Audio time series (amplitude values between -1 and 1)
    sr : int
        Sample rate of the returned audio
    
    LEARNING:
    ---------
    The returned array 'y' contains amplitude values over time.
    Each value represents the air pressure displacement at that instant.
    
    Example: y = [0.1, 0.3, 0.5, 0.3, 0.1, -0.1, -0.3, ...]
    
    Duration in seconds = len(y) / sr
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    # Load with librosa (handles many formats, automatic resampling)
    y, sr_loaded = librosa.load(
        filepath,
        sr=sr,
        mono=mono,
        duration=duration,
        offset=offset
    )
    
    # Normalize to [-1, 1] range if needed
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val
        warnings.warn(f"Audio normalized from max amplitude {max_val:.2f}")
    
    return y, sr


def save_audio(
    filepath: Union[str, Path],
    y: np.ndarray,
    sr: int = 22050,
    normalize: bool = True
) -> None:
    """
    Save audio array to a file.
    
    Parameters
    ----------
    filepath : str or Path
        Output path. Format determined by extension (.wav recommended)
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    normalize : bool, default=True
        If True, normalize audio to prevent clipping
    
    LEARNING:
    ---------
    WAV files store audio as uncompressed samples.
    MP3 uses lossy compression (smaller files, some quality loss).
    For processing, always use WAV to avoid compression artifacts.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.95  # Leave headroom to prevent clipping
    
    sf.write(filepath, y, sr)
    print(f"Saved audio to {filepath} ({len(y)/sr:.2f}s @ {sr}Hz)")


def play_audio(y: np.ndarray, sr: int = 22050) -> None:
    """
    Play audio in a Jupyter notebook.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    
    LEARNING:
    ---------
    IPython.display.Audio creates an interactive audio player widget.
    This only works in Jupyter notebooks, not in regular Python scripts.
    """
    try:
        from IPython.display import Audio, display
        display(Audio(y, rate=sr))
    except ImportError:
        print("Audio playback requires IPython (Jupyter notebook)")
        print(f"Audio duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")


def generate_test_signals(sr: int = 22050, duration: float = 3.0) -> dict:
    """
    Generate synthetic test signals for experimentation.
    
    Useful when you don't have audio files yet!
    
    Returns
    -------
    dict : Dictionary containing various test signals
    
    LEARNING:
    ---------
    Pure tones are sine waves at specific frequencies.
    White noise has equal energy at all frequencies.
    These simple signals help us understand transformations.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    signals = {}
    
    # Pure sine wave (A4 = 440 Hz)
    signals['sine_440'] = np.sin(2 * np.pi * 440 * t)
    
    # Melody: simple ascending scale
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major
    melody = np.zeros_like(t)
    note_duration = duration / len(freqs)
    for i, freq in enumerate(freqs):
        start_idx = int(i * note_duration * sr)
        end_idx = int((i + 1) * note_duration * sr)
        melody[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        # Apply envelope to each note
        envelope = np.exp(-3 * (t[start_idx:end_idx] - t[start_idx]))
        melody[start_idx:end_idx] *= envelope
    signals['melody'] = melody
    
    # White noise (simulates ambient/texture sounds)
    signals['white_noise'] = np.random.randn(len(t)) * 0.3
    
    # Filtered noise (more like wind/clouds - low frequency emphasis)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 1000 / (sr/2), btype='low')
    signals['wind_noise'] = filtfilt(b, a, signals['white_noise'])
    
    # Pink noise (1/f noise - more natural sounding)
    white = np.random.randn(len(t))
    # Simple approximation of pink noise
    pink = np.cumsum(white) / 100
    pink = pink - np.mean(pink)
    pink = pink / np.max(np.abs(pink)) * 0.5
    signals['pink_noise'] = pink
    
    # Chirp (frequency sweep - good for testing)
    f0, f1 = 100, 2000
    signals['chirp'] = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    
    return signals


def get_audio_info(y: np.ndarray, sr: int) -> dict:
    """
    Get basic information about an audio signal.
    
    LEARNING:
    ---------
    These metrics help us understand the audio before processing:
    - Duration: how long the audio is
    - RMS: root-mean-square, a measure of average loudness
    - Peak: maximum amplitude (important for avoiding clipping)
    - Dynamic range: difference between loudest and quietest parts
    """
    duration = len(y) / sr
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    
    # Dynamic range in dB
    # Add small epsilon to avoid log(0)
    y_abs = np.abs(y) + 1e-10
    db = 20 * np.log10(y_abs)
    dynamic_range = np.max(db) - np.percentile(db, 10)
    
    return {
        'duration_seconds': duration,
        'sample_rate': sr,
        'num_samples': len(y),
        'rms': rms,
        'peak': peak,
        'dynamic_range_db': dynamic_range
    }

