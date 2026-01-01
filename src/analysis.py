"""
Audio Analysis Module

Extract meaningful representations from audio signals.

LEARNING NOTES:
- STFT (Short-Time Fourier Transform) converts audio to time-frequency representation
- Pitch detection finds the fundamental frequency (F0) over time
- Envelope captures the amplitude dynamics (loud/soft patterns)
- These representations let us separate "what" (content) from "how" (style)
"""

import numpy as np
import librosa
from scipy.signal import hilbert
from typing import Tuple, Optional
import warnings


def compute_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of an audio signal.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    n_fft : int, default=2048
        FFT window size. Larger = better frequency resolution, worse time resolution.
        2048 samples ≈ 93ms at 22050 Hz
    hop_length : int, default=512
        Number of samples between successive frames.
        Smaller = more overlap = smoother spectrogram
    win_length : int, optional
        Window length. Defaults to n_fft.
    window : str, default='hann'
        Window function. 'hann' is standard - tapers edges to reduce artifacts.
    
    Returns
    -------
    S : np.ndarray (complex)
        Complex STFT matrix. Shape: (n_frequencies, n_frames)
        Magnitude = |S| gives the spectrogram
        Phase = angle(S) needed for reconstruction
    frequencies : np.ndarray
        Frequency values for each bin
    
    LEARNING:
    ---------
    The STFT is the heart of audio processing. It answers:
    "What frequencies are present at each moment in time?"
    
    Trade-off: n_fft controls frequency vs time resolution
    - Large n_fft: See individual harmonics, but blur fast changes
    - Small n_fft: Track fast changes, but blur close frequencies
    
    The spectrogram |S|² shows energy distribution over time-frequency.
    """
    if win_length is None:
        win_length = n_fft
    
    # Compute STFT
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    
    # Frequency axis
    frequencies = librosa.fft_frequencies(sr=22050, n_fft=n_fft)
    
    return S, frequencies


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Compute mel-scaled spectrogram.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    n_mels : int, default=128
        Number of mel bands
    n_fft : int
        FFT window size
    hop_length : int
        Hop size
    fmin : float
        Minimum frequency
    fmax : float, optional
        Maximum frequency (defaults to sr/2)
    
    Returns
    -------
    S_mel : np.ndarray
        Mel spectrogram. Shape: (n_mels, n_frames)
    
    LEARNING:
    ---------
    The mel scale matches human pitch perception:
    - We hear differences between 100-200 Hz more than 5000-5100 Hz
    - Mel bands are spaced logarithmically in frequency
    - This makes the representation more perceptually meaningful
    
    Mel spectrograms are often used as input to neural networks.
    """
    if fmax is None:
        fmax = sr / 2
    
    S_mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    return S_mel


def extract_pitch(
    y: np.ndarray,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    hop_length: int = 512,
    method: str = 'pyin'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pitch (fundamental frequency) contour from audio.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    fmin : float
        Minimum expected frequency (Hz). 50 Hz covers bass voices.
    fmax : float
        Maximum expected frequency (Hz). 2000 Hz covers most singing.
    hop_length : int
        Hop size for analysis
    method : str
        'pyin' (probabilistic YIN) or 'yin'
    
    Returns
    -------
    f0 : np.ndarray
        Fundamental frequency in Hz for each frame. NaN where unvoiced.
    voiced_flag : np.ndarray
        Boolean array indicating voiced frames
    times : np.ndarray
        Time stamps for each frame
    
    LEARNING:
    ---------
    Pitch is the perceived frequency of a sound - the note being sung.
    
    F0 (fundamental frequency) is the lowest frequency component in a 
    harmonic sound. When Elvis sings an A4, F0 ≈ 440 Hz.
    
    "Voiced" means there's a clear pitch (singing/speech).
    "Unvoiced" means no clear pitch (breathing, consonants, silence).
    
    The pitch contour is the MELODY - this is what we want to transfer
    to make the clouds "sing"!
    """
    if method == 'pyin':
        # PYIN: Probabilistic YIN - handles pitch uncertainty well
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )
    else:
        # Basic YIN
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )
        voiced_flag = ~np.isnan(f0)
    
    # Time stamps for each frame
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # Replace NaN with 0 for unvoiced regions (easier to work with)
    f0_clean = np.nan_to_num(f0, nan=0.0)
    
    return f0_clean, voiced_flag, times


def extract_envelope(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    method: str = 'rms'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitude envelope from audio.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop size
    method : str
        'rms' (root mean square) or 'hilbert' (analytic envelope)
    
    Returns
    -------
    envelope : np.ndarray
        Amplitude envelope over time
    times : np.ndarray
        Time stamps
    
    LEARNING:
    ---------
    The envelope captures the "shape" of the sound's loudness over time.
    
    Think of it as tracing the outline of the waveform:
    - Attack: How quickly the sound starts
    - Decay: Initial fall after attack
    - Sustain: Steady-state level
    - Release: How the sound fades away
    
    When Elvis sings a phrase, the envelope shows where he breathes,
    emphasizes words, and phrases the melody. This RHYTHM is part of 
    the style we want to transfer!
    """
    if method == 'rms':
        # RMS envelope - standard approach
        envelope = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    elif method == 'hilbert':
        # Hilbert envelope - smoother, captures instantaneous amplitude
        analytic = hilbert(y)
        envelope_full = np.abs(analytic)
        # Downsample to match hop_length
        n_frames = len(y) // hop_length
        envelope = np.array([
            np.mean(envelope_full[i*hop_length:(i+1)*hop_length])
            for i in range(n_frames)
        ])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    times = librosa.times_like(envelope, sr=sr, hop_length=hop_length)
    
    return envelope, times


def extract_spectral_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> dict:
    """
    Extract various spectral features useful for style characterization.
    
    Returns
    -------
    features : dict
        Dictionary of spectral features over time
    
    LEARNING:
    ---------
    These features describe the "color" or "timbre" of sound:
    
    - Spectral Centroid: "Brightness" - higher = brighter/sharper sound
    - Spectral Bandwidth: How spread out the frequencies are
    - Spectral Rolloff: Frequency below which 85% of energy lies
    - Spectral Flatness: How noise-like vs tonal (1=noise, 0=tone)
    - Zero Crossing Rate: How often signal crosses zero (correlates with noisiness)
    
    Elvis's voice has characteristic spectral features that distinguish
    it from other singers. Environmental sounds have different features.
    """
    features = {}
    
    # Spectral centroid - "center of mass" of spectrum
    features['centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]
    
    # Spectral bandwidth - spread around centroid
    features['bandwidth'] = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=hop_length
    )[0]
    
    # Spectral rolloff - frequency below which 85% of energy
    features['rolloff'] = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length
    )[0]
    
    # Spectral flatness - tonality measure
    features['flatness'] = librosa.feature.spectral_flatness(
        y=y, hop_length=hop_length
    )[0]
    
    # Zero crossing rate
    features['zcr'] = librosa.feature.zero_crossing_rate(
        y, hop_length=hop_length
    )[0]
    
    # Time axis
    features['times'] = librosa.times_like(
        features['centroid'], sr=sr, hop_length=hop_length
    )
    
    return features


def extract_formants_approx(
    y: np.ndarray,
    sr: int = 22050,
    n_formants: int = 4,
    hop_length: int = 512
) -> np.ndarray:
    """
    Approximate formant extraction using LPC (Linear Predictive Coding).
    
    Parameters
    ----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    n_formants : int
        Number of formants to extract
    hop_length : int
        Hop size
    
    Returns
    -------
    formants : np.ndarray
        Formant frequencies. Shape: (n_formants, n_frames)
    
    LEARNING:
    ---------
    Formants are resonant frequencies of the vocal tract.
    They determine vowel sounds:
    
    - F1 (first formant): Related to tongue height (250-900 Hz)
      High F1 = open vowel (like "ah"), Low F1 = closed ("ee")
    
    - F2 (second formant): Related to tongue position (700-2500 Hz)
      High F2 = front vowel ("ee"), Low F2 = back vowel ("oo")
    
    - F3, F4: Add individual voice character
    
    Formants give voices their recognizable quality. Adding formant-like
    resonances to the clouds could make them sound more "vocal"!
    
    Note: This is a simplified approximation. Professional formant 
    extraction is more complex.
    """
    from scipy.signal import lfilter, hamming
    from scipy.linalg import solve_toeplitz
    
    # LPC order (rule of thumb: 2 + formants*2)
    order = 2 + n_formants * 2
    
    # Frame the signal
    frames = librosa.util.frame(y, frame_length=2048, hop_length=hop_length)
    n_frames = frames.shape[1]
    
    formants = np.zeros((n_formants, n_frames))
    
    for i in range(n_frames):
        frame = frames[:, i] * hamming(len(frames[:, i]))
        
        # Compute autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Levinson-Durbin recursion for LPC coefficients
        try:
            # Solve Toeplitz system for LPC coefficients
            r = autocorr[:order+1]
            a = solve_toeplitz((r[:-1], r[:-1]), r[1:])
            a = np.concatenate([[1], -a])
            
            # Find roots of LPC polynomial
            roots = np.roots(a)
            
            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * sr / (2 * np.pi)
            
            # Keep only positive frequencies, sorted
            freqs = freqs[freqs > 0]
            freqs = np.sort(freqs)
            
            # Take first n_formants
            if len(freqs) >= n_formants:
                formants[:, i] = freqs[:n_formants]
            else:
                formants[:len(freqs), i] = freqs
                
        except Exception:
            # If extraction fails, use previous frame or zeros
            if i > 0:
                formants[:, i] = formants[:, i-1]
    
    return formants


def decompose_svd(
    S: np.ndarray,
    n_components: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose spectrogram using Singular Value Decomposition.
    
    Parameters
    ----------
    S : np.ndarray
        Spectrogram magnitude (frequency x time)
    n_components : int
        Number of components to keep
    
    Returns
    -------
    U : np.ndarray
        Left singular vectors (frequency patterns). Shape: (n_freq, n_components)
    s : np.ndarray
        Singular values (importance of each component)
    Vt : np.ndarray
        Right singular vectors (time activations). Shape: (n_components, n_time)
    
    LEARNING:
    ---------
    SVD decomposes the spectrogram into:
    
    S ≈ U @ diag(s) @ Vt
    
    - U columns: "Spectral templates" - frequency patterns
    - s: How important each template is (larger = more important)
    - Vt rows: "Activations" - when each template is active
    
    This is like finding the building blocks of the sound!
    
    For clouds/ambient sounds, the first few components capture
    the main texture. We can modify or replace these to change
    the character of the sound.
    
    This is the LINEAR ALGEBRA part of content extraction!
    """
    from scipy.linalg import svd
    
    # Compute SVD
    U, s, Vt = svd(S, full_matrices=False)
    
    # Keep only n_components
    U = U[:, :n_components]
    s = s[:n_components]
    Vt = Vt[:n_components, :]
    
    # Print explained variance
    total_var = np.sum(s**2)
    explained = np.cumsum(s**2) / total_var
    print(f"SVD: {n_components} components explain {explained[-1]*100:.1f}% of variance")
    
    return U, s, Vt


def decompose_pca(
    S: np.ndarray,
    n_components: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose spectrogram using PCA (mean-centered SVD).
    
    Similar to SVD but centers the data first.
    
    LEARNING:
    ---------
    PCA finds the directions of maximum variance in the data.
    
    For spectrograms:
    - First PC: Most common spectral shape
    - Second PC: Largest deviation from the mean
    - etc.
    
    PCA is useful for understanding what makes a sound distinctive.
    """
    from sklearn.decomposition import PCA
    
    # PCA expects (samples, features), we have (features, samples)
    # Transpose so each time frame is a sample
    S_T = S.T
    
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(S_T)
    
    components = pca.components_  # Shape: (n_components, n_freq)
    explained_variance = pca.explained_variance_ratio_
    
    print(f"PCA: {n_components} components explain {sum(explained_variance)*100:.1f}% of variance")
    
    # Return in same format as SVD for consistency
    return components.T, np.sqrt(pca.explained_variance_), transformed.T

