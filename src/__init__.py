"""
Audio Style Projection - Core Library

Transform environmental sounds using vocal characteristics.
"""

from .audio_io import load_audio, save_audio, play_audio
from .analysis import compute_stft, compute_mel_spectrogram, extract_pitch, extract_envelope
from .transform import spectral_modulation, envelope_transfer, style_projection
from .synthesis import reconstruct_audio, apply_vocoder
from .visualization import plot_waveform, plot_spectrogram, plot_pitch_contour

__version__ = "0.1.0"
__all__ = [
    "load_audio",
    "save_audio", 
    "play_audio",
    "compute_stft",
    "compute_mel_spectrogram",
    "extract_pitch",
    "extract_envelope",
    "spectral_modulation",
    "envelope_transfer",
    "style_projection",
    "reconstruct_audio",
    "apply_vocoder",
    "plot_waveform",
    "plot_spectrogram",
    "plot_pitch_contour",
]

