"""
Audio Synthesis Module

Convert spectral representations back to audio.

LEARNING NOTES:
- Inverse STFT reconstructs audio from complex spectrogram
- Phase is crucial for natural-sounding reconstruction
- Neural vocoders can generate high-quality audio from mel spectrograms
"""

import numpy as np
import librosa
from typing import Optional
import warnings


def reconstruct_audio(
    S: np.ndarray,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = 'hann',
    n_iter: int = 32
) -> np.ndarray:
    """
    Reconstruct audio from STFT representation.
    
    Parameters
    ----------
    S : np.ndarray
        STFT matrix. Can be:
        - Complex: Use magnitude and phase directly
        - Real (magnitude only): Use Griffin-Lim for phase estimation
    hop_length : int
        Hop size (must match analysis)
    win_length : int, optional
        Window length
    window : str
        Window function
    n_iter : int
        Griffin-Lim iterations (only if phase estimation needed)
    
    Returns
    -------
    y : np.ndarray
        Reconstructed audio time series
    
    LEARNING:
    ---------
    The inverse STFT is straightforward IF you have phase information:
    
    y = ISTFT(magnitude * exp(i * phase))
    
    But if you only have magnitude (spectrogram), phase must be estimated.
    Griffin-Lim is an iterative algorithm that finds consistent phases:
    
    1. Start with random phases
    2. ISTFT to get audio
    3. STFT the audio
    4. Keep new phases, restore target magnitude
    5. Repeat
    
    More iterations = better quality, but slower.
    """
    # Check if S is complex (has phase) or real (magnitude only)
    if np.iscomplexobj(S):
        # Direct inverse STFT
        y = librosa.istft(
            S,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )
    else:
        # Phase estimation using Griffin-Lim
        warnings.warn("No phase information - using Griffin-Lim reconstruction")
        y = librosa.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )
    
    return y


def reconstruct_from_mel(
    S_mel: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 64
) -> np.ndarray:
    """
    Reconstruct audio from mel spectrogram.
    
    Parameters
    ----------
    S_mel : np.ndarray
        Mel spectrogram (n_mels x n_frames)
    sr : int
        Sample rate
    n_fft : int
        FFT size
    hop_length : int
        Hop size
    n_iter : int
        Griffin-Lim iterations
    
    Returns
    -------
    y : np.ndarray
        Reconstructed audio
    
    LEARNING:
    ---------
    Mel spectrograms lose information (they're a compressed representation).
    Inversion requires:
    
    1. Convert mel to linear spectrogram (approximate inverse of mel filterbank)
    2. Estimate phases with Griffin-Lim
    
    Quality is lower than inverting a full STFT, but mel spectrograms
    are useful for neural network processing.
    """
    # Invert mel filterbank
    S_linear = librosa.feature.inverse.mel_to_stft(
        S_mel,
        sr=sr,
        n_fft=n_fft
    )
    
    # Griffin-Lim phase reconstruction
    y = librosa.griffinlim(
        S_linear,
        n_iter=n_iter,
        hop_length=hop_length
    )
    
    return y


def apply_vocoder(
    S_mel: np.ndarray,
    sr: int = 22050,
    vocoder: str = 'griffin_lim'
) -> np.ndarray:
    """
    Apply a vocoder to convert mel spectrogram to audio.
    
    Parameters
    ----------
    S_mel : np.ndarray
        Mel spectrogram
    sr : int
        Sample rate
    vocoder : str
        Vocoder to use:
        - 'griffin_lim': Basic iterative phase estimation
        - 'neural': (placeholder for HiFi-GAN, etc.)
    
    Returns
    -------
    y : np.ndarray
        Synthesized audio
    
    LEARNING:
    ---------
    Neural vocoders (HiFi-GAN, WaveGlow, etc.) produce much higher
    quality audio than Griffin-Lim because they learn to generate
    realistic waveforms from spectrograms.
    
    However, they require:
    - Pre-trained model weights
    - GPU for reasonable speed
    - Matching mel spectrogram configuration
    
    For learning purposes, Griffin-Lim is sufficient and shows
    the principles clearly.
    """
    if vocoder == 'griffin_lim':
        return reconstruct_from_mel(S_mel, sr)
    
    elif vocoder == 'neural':
        # Placeholder for neural vocoder integration
        # Would require: import torch, load pretrained HiFi-GAN, etc.
        warnings.warn("Neural vocoder not implemented, falling back to Griffin-Lim")
        return reconstruct_from_mel(S_mel, sr)
    
    else:
        raise ValueError(f"Unknown vocoder: {vocoder}")


def crossfade_audio(
    y1: np.ndarray,
    y2: np.ndarray,
    crossfade_samples: int = 1000
) -> np.ndarray:
    """
    Crossfade between two audio segments.
    
    LEARNING:
    ---------
    Crossfading smoothly transitions between audio segments.
    
    y_out = y1 * fade_out + y2 * fade_in
    
    where fade_out goes from 1 to 0 and fade_in goes from 0 to 1.
    
    This prevents clicks and pops at segment boundaries.
    """
    # Ensure same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Create fade curves
    crossfade_samples = min(crossfade_samples, min_len // 2)
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Apply crossfade at the end/beginning
    result = y1.copy()
    transition_point = min_len - crossfade_samples
    
    for i in range(crossfade_samples):
        idx = transition_point + i
        result[idx] = y1[idx] * fade_out[i] + y2[i] * fade_in[i]
    
    return result


def normalize_audio(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to a target peak level in dB.
    
    LEARNING:
    ---------
    Audio is often normalized before saving to:
    1. Maximize signal-to-noise ratio
    2. Prevent clipping (values > 1.0)
    3. Maintain consistent loudness
    
    -3 dB peak leaves some headroom for any downstream processing.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        target_linear = 10 ** (target_db / 20)
        y = y * (target_linear / peak)
    return y


def apply_reverb(
    y: np.ndarray,
    sr: int = 22050,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.3
) -> np.ndarray:
    """
    Apply simple reverb effect to audio.
    
    Parameters
    ----------
    y : np.ndarray
        Input audio
    sr : int
        Sample rate
    room_size : float [0, 1]
        Size of simulated room (larger = longer reverb)
    damping : float [0, 1]
        High frequency damping (higher = less bright reverb)
    wet_level : float [0, 1]
        Mix of reverb signal
    
    LEARNING:
    ---------
    Reverb makes audio sound like it's in a physical space.
    
    Simple reverb uses delay lines with feedback:
    1. Delay the signal by various amounts
    2. Feed back the delayed signal (with attenuation)
    3. Apply filtering for damping
    
    Reverb can help blend transformed audio more naturally!
    """
    from scipy.signal import lfilter
    
    # Simple Schroeder reverb using comb and allpass filters
    
    # Comb filter delays (in samples)
    comb_delays = [int(sr * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
    comb_delays = [int(d * room_size) for d in comb_delays]
    
    # Feedback gains
    feedback = 0.8 * room_size
    
    # Initialize output
    wet = np.zeros_like(y)
    
    # Apply parallel comb filters
    for delay in comb_delays:
        if delay < len(y):
            b = np.zeros(delay + 1)
            b[0] = 1
            a = np.zeros(delay + 1)
            a[0] = 1
            a[-1] = -feedback
            
            comb_out = lfilter(b, a, y)
            
            # Apply damping (simple lowpass)
            if damping > 0:
                alpha = 0.5 * damping
                b_lp = [1 - alpha]
                a_lp = [1, -alpha]
                comb_out = lfilter(b_lp, a_lp, comb_out)
            
            wet += comb_out
    
    # Normalize wet signal
    wet = wet / 4
    
    # Mix dry and wet
    y_reverb = (1 - wet_level) * y + wet_level * wet
    
    return y_reverb

