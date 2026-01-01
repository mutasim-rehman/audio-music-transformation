"""
Audio Transformation Module

Core algorithms for style projection and audio transformation.

LEARNING NOTES:
This is where the magic happens! We take:
- Content from source (texture of clouds)
- Style from target (melody/rhythm of Elvis)
And blend them into something new.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Optional, Union
import warnings


def spectral_modulation(
    S_source: np.ndarray,
    f0_target: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    modulation_strength: float = 0.5,
    bandwidth_hz: float = 100.0
) -> np.ndarray:
    """
    Modulate source spectrogram based on target pitch contour.
    
    This makes the source audio "follow" the target melody by boosting
    frequencies near the target pitch at each moment.
    
    Parameters
    ----------
    S_source : np.ndarray (complex)
        Source STFT matrix (frequency x time)
    f0_target : np.ndarray
        Target pitch contour (Hz) - one value per frame
    sr : int
        Sample rate
    n_fft : int
        FFT size used to create S_source
    modulation_strength : float [0, 1]
        How strongly to apply the modulation
        0 = no change, 1 = full modulation
    bandwidth_hz : float
        Bandwidth of the modulation filter (Hz)
        Larger = affects more frequencies around the pitch
    
    Returns
    -------
    S_modulated : np.ndarray (complex)
        Modulated STFT matrix
    
    LEARNING:
    ---------
    The key insight: We don't replace the source audio with the target.
    Instead, we EMPHASIZE parts of the source that align with the target pitch.
    
    Imagine the cloud sound contains all frequencies (it's noise-like).
    When Elvis sings an A (440 Hz), we boost the 440 Hz region of the clouds.
    The cloud texture is still there, but now it has a pitch!
    
    This is like shining a spotlight through fog - the fog is still fog,
    but you can see patterns in where the light falls.
    """
    # Get magnitude and phase
    magnitude = np.abs(S_source)
    phase = np.angle(S_source)
    
    n_freq, n_frames = magnitude.shape
    
    # Frequency bins
    frequencies = np.linspace(0, sr / 2, n_freq)
    
    # Ensure f0_target matches number of frames
    if len(f0_target) != n_frames:
        # Resample pitch contour to match
        f0_target = resample(f0_target, n_frames)
    
    # Create modulation mask
    modulation_mask = np.ones_like(magnitude)
    
    for t in range(n_frames):
        f0 = f0_target[t]
        
        if f0 > 0:  # Only modulate where there's a pitch
            # Create Gaussian bump centered at f0 and its harmonics
            for harmonic in range(1, 8):  # Include harmonics for richer sound
                center_freq = f0 * harmonic
                if center_freq > sr / 2:
                    break
                
                # Gaussian modulation centered at harmonic
                sigma = bandwidth_hz / 2
                harmonic_strength = 1.0 / harmonic  # Harmonics get weaker
                
                gaussian = np.exp(-0.5 * ((frequencies - center_freq) / sigma) ** 2)
                modulation_mask[:, t] += modulation_strength * harmonic_strength * gaussian
    
    # Normalize modulation to prevent excessive amplification
    modulation_mask = modulation_mask / np.max(modulation_mask)
    
    # Apply modulation to magnitude
    magnitude_modulated = magnitude * modulation_mask
    
    # Reconstruct complex STFT
    S_modulated = magnitude_modulated * np.exp(1j * phase)
    
    return S_modulated


def envelope_transfer(
    y_source: np.ndarray,
    envelope_target: np.ndarray,
    times_target: np.ndarray,
    sr: int = 22050,
    transfer_strength: float = 0.5,
    preserve_dynamics: bool = True
) -> np.ndarray:
    """
    Transfer amplitude envelope from target to source.
    
    Parameters
    ----------
    y_source : np.ndarray
        Source audio time series
    envelope_target : np.ndarray
        Target amplitude envelope
    times_target : np.ndarray
        Time stamps for envelope
    sr : int
        Sample rate
    transfer_strength : float [0, 1]
        Blend between source and target envelope
    preserve_dynamics : bool
        If True, preserve relative dynamics of source
    
    Returns
    -------
    y_shaped : np.ndarray
        Source audio with modified envelope
    
    LEARNING:
    ---------
    The envelope carries the RHYTHM and PHRASING of the performance.
    
    When Elvis sings "Wise men say...", he shapes each phrase:
    - Emphasis on certain words
    - Breaths between phrases
    - Dynamic swells
    
    By transferring this envelope to clouds, they will "breathe" and
    "phrase" like Elvis, even though they're still cloud sounds!
    
    This is amplitude modulation - multiplying the signal by a varying gain.
    """
    # Interpolate target envelope to match source length
    duration_source = len(y_source) / sr
    times_source = np.linspace(0, duration_source, len(y_source))
    
    # Extend target envelope if needed
    if times_target[-1] < duration_source:
        # Repeat the envelope pattern
        n_repeats = int(np.ceil(duration_source / times_target[-1]))
        envelope_target = np.tile(envelope_target, n_repeats)
        times_target = np.concatenate([
            times_target + i * times_target[-1] 
            for i in range(n_repeats)
        ])
    
    # Interpolate to source sample rate
    interpolator = interp1d(
        times_target[:len(envelope_target)],
        envelope_target,
        kind='linear',
        fill_value='extrapolate'
    )
    envelope_resampled = interpolator(times_source)
    
    # Normalize envelope
    envelope_resampled = envelope_resampled / (np.max(envelope_resampled) + 1e-10)
    
    if preserve_dynamics:
        # Compute source envelope
        from .analysis import extract_envelope
        env_source, _ = extract_envelope(y_source, sr)
        
        # Interpolate source envelope to full resolution
        times_env = np.linspace(0, duration_source, len(env_source))
        interp_source = interp1d(times_env, env_source, fill_value='extrapolate')
        env_source_full = interp_source(times_source)
        env_source_full = env_source_full / (np.max(env_source_full) + 1e-10)
        
        # Blend envelopes
        envelope_final = (
            transfer_strength * envelope_resampled + 
            (1 - transfer_strength) * env_source_full
        )
    else:
        # Direct envelope replacement
        envelope_final = (
            transfer_strength * envelope_resampled + 
            (1 - transfer_strength)
        )
    
    # Apply envelope to source
    y_shaped = y_source * envelope_final
    
    return y_shaped


def style_projection(
    S_source: np.ndarray,
    S_target: np.ndarray,
    alpha: float = 0.5,
    method: str = 'magnitude_blend'
) -> np.ndarray:
    """
    Project source spectrogram toward target style.
    
    Parameters
    ----------
    S_source : np.ndarray (complex)
        Source STFT
    S_target : np.ndarray (complex)
        Target STFT
    alpha : float [0, 1]
        Blending parameter
        0 = pure source
        1 = maximum target influence
    method : str
        Projection method:
        - 'magnitude_blend': Blend magnitudes, keep source phase
        - 'spectral_transfer': Transfer spectral envelope
        - 'svd_project': Project in SVD space
    
    Returns
    -------
    S_projected : np.ndarray (complex)
        Projected STFT
    
    LEARNING:
    ---------
    This is where we implement the core equation:
    
    x_output = α·Style(x_source, x_target) + (1-α)·x_source
    
    Different methods capture different aspects of "style":
    
    1. Magnitude Blend: Simply interpolate spectral magnitudes
       - Fast and predictable
       - May sound unnatural at high α
    
    2. Spectral Transfer: Match spectral envelope (overall shape)
       - Preserves fine structure of source
       - Transfers "color" of target
    
    3. SVD Project: Work in the principal component space
       - More sophisticated
       - Can capture complex patterns
    """
    # Get magnitudes and phases
    mag_source = np.abs(S_source)
    phase_source = np.angle(S_source)
    mag_target = np.abs(S_target)
    
    # Ensure same shape (truncate to shorter)
    min_frames = min(mag_source.shape[1], mag_target.shape[1])
    mag_source = mag_source[:, :min_frames]
    phase_source = phase_source[:, :min_frames]
    mag_target = mag_target[:, :min_frames]
    
    if method == 'magnitude_blend':
        # Simple magnitude interpolation
        mag_projected = (1 - alpha) * mag_source + alpha * mag_target
        
    elif method == 'spectral_transfer':
        # Transfer spectral envelope while preserving fine structure
        
        # Compute spectral envelopes (smoothed magnitude)
        from scipy.ndimage import uniform_filter1d
        env_source = uniform_filter1d(mag_source, size=20, axis=0)
        env_target = uniform_filter1d(mag_target, size=20, axis=0)
        
        # Fine structure = original / envelope
        fine_source = mag_source / (env_source + 1e-10)
        
        # Blend envelopes
        env_blended = (1 - alpha) * env_source + alpha * env_target
        
        # Reconstruct with blended envelope and source fine structure
        mag_projected = fine_source * env_blended
        
    elif method == 'svd_project':
        # Project in SVD space
        from .analysis import decompose_svd
        
        # Decompose both
        U_source, s_source, Vt_source = decompose_svd(mag_source, n_components=20)
        U_target, s_target, Vt_target = decompose_svd(mag_target, n_components=20)
        
        # Blend components
        U_blend = (1 - alpha) * U_source + alpha * U_target
        s_blend = (1 - alpha) * s_source + alpha * s_target
        
        # Reconstruct with source temporal structure
        mag_projected = U_blend @ np.diag(s_blend) @ Vt_source
        mag_projected = np.maximum(mag_projected, 0)  # Ensure non-negative
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reconstruct complex STFT with source phase
    S_projected = mag_projected * np.exp(1j * phase_source)
    
    return S_projected


def harmonic_transfer(
    S_source: np.ndarray,
    f0_target: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    n_harmonics: int = 10,
    transfer_strength: float = 0.5
) -> np.ndarray:
    """
    Transfer harmonic structure from target to source.
    
    Creates a harmonic comb filter based on target pitch and applies
    it to the source spectrogram.
    
    LEARNING:
    ---------
    Harmonic sounds (singing, instruments) have energy at integer
    multiples of the fundamental:
    
    F0 = 440 Hz (A4)
    H2 = 880 Hz
    H3 = 1320 Hz
    H4 = 1760 Hz
    ...
    
    Environmental sounds are usually "inharmonic" - energy spread randomly.
    
    By creating a harmonic comb filter from the target pitch and
    applying it to the source, we impose harmonic structure on
    non-harmonic sounds. This creates a "pitched" quality while
    keeping the original texture.
    """
    magnitude = np.abs(S_source)
    phase = np.angle(S_source)
    n_freq, n_frames = magnitude.shape
    
    frequencies = np.linspace(0, sr / 2, n_freq)
    
    # Resample f0 if needed
    if len(f0_target) != n_frames:
        f0_target = resample(f0_target, n_frames)
    
    # Create harmonic comb filter for each frame
    comb_filter = np.ones_like(magnitude) * (1 - transfer_strength)
    
    for t in range(n_frames):
        f0 = f0_target[t]
        
        if f0 > 0:
            # Add peaks at each harmonic
            for h in range(1, n_harmonics + 1):
                harmonic_freq = f0 * h
                if harmonic_freq > sr / 2:
                    break
                
                # Find closest frequency bin
                bin_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                
                # Create peak with falloff
                bandwidth = max(3, int(n_freq * 0.01))  # Adaptive bandwidth
                for offset in range(-bandwidth, bandwidth + 1):
                    idx = bin_idx + offset
                    if 0 <= idx < n_freq:
                        # Triangular window
                        weight = 1 - abs(offset) / (bandwidth + 1)
                        comb_filter[idx, t] += transfer_strength * weight / h
    
    # Normalize
    comb_filter = np.clip(comb_filter, 0, 2)
    
    # Apply filter
    magnitude_filtered = magnitude * comb_filter
    
    return magnitude_filtered * np.exp(1j * phase)


def formant_imposition(
    S_source: np.ndarray,
    formant_freqs: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    bandwidth_hz: float = 150.0,
    strength: float = 0.3
) -> np.ndarray:
    """
    Impose formant-like resonances on source spectrogram.
    
    Parameters
    ----------
    S_source : np.ndarray
        Source STFT
    formant_freqs : np.ndarray
        Formant frequencies. Shape: (n_formants,) or (n_formants, n_frames)
    sr : int
        Sample rate
    n_fft : int
        FFT size
    bandwidth_hz : float
        Formant bandwidth
    strength : float [0, 1]
        How strongly to apply formants
    
    LEARNING:
    ---------
    Formants are what make vowels sound like vowels.
    
    By adding formant resonances to environmental sounds, we can
    make them sound more "vocal" - as if a voice is hidden in the
    sound trying to speak.
    
    This is subtle but powerful for creating the "humming" quality.
    """
    magnitude = np.abs(S_source)
    phase = np.angle(S_source)
    n_freq, n_frames = magnitude.shape
    
    frequencies = np.linspace(0, sr / 2, n_freq)
    
    # Handle static vs time-varying formants
    if formant_freqs.ndim == 1:
        formant_freqs = np.tile(formant_freqs[:, np.newaxis], (1, n_frames))
    
    # Create formant filter
    formant_filter = np.ones_like(magnitude)
    
    n_formants = formant_freqs.shape[0]
    
    for t in range(n_frames):
        for f_idx in range(n_formants):
            f_center = formant_freqs[f_idx, min(t, formant_freqs.shape[1]-1)]
            
            if f_center > 0 and f_center < sr / 2:
                # Formant resonance (Gaussian peak)
                sigma = bandwidth_hz / 2
                resonance = np.exp(-0.5 * ((frequencies - f_center) / sigma) ** 2)
                formant_filter[:, t] += strength * resonance
    
    # Apply formant filter
    magnitude_formatted = magnitude * formant_filter
    
    return magnitude_formatted * np.exp(1j * phase)


def create_singing_texture(
    y_source: np.ndarray,
    y_target: np.ndarray,
    sr: int = 22050,
    alpha: float = 0.5,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, dict]:
    """
    High-level function: Transform source texture to "sing" like target.
    
    This combines all the transformation techniques:
    1. Pitch extraction from target
    2. Envelope extraction from target
    3. Spectral modulation of source
    4. Envelope transfer
    5. Optional formant imposition
    
    Parameters
    ----------
    y_source : np.ndarray
        Source audio (environmental sound)
    y_target : np.ndarray
        Target audio (vocal performance)
    sr : int
        Sample rate
    alpha : float [0, 1]
        Overall transformation strength
    n_fft : int
        FFT size
    hop_length : int
        Hop size
    
    Returns
    -------
    y_output : np.ndarray
        Transformed audio
    info : dict
        Intermediate results for visualization
    """
    from .analysis import compute_stft, extract_pitch, extract_envelope
    from .synthesis import reconstruct_audio
    
    info = {}
    
    # 1. Analyze target
    print("Extracting pitch from target...")
    f0_target, voiced_target, times_target = extract_pitch(y_target, sr, hop_length=hop_length)
    info['f0_target'] = f0_target
    info['voiced_target'] = voiced_target
    
    print("Extracting envelope from target...")
    env_target, env_times = extract_envelope(y_target, sr, hop_length=hop_length)
    info['envelope_target'] = env_target
    
    # 2. Compute source STFT
    print("Computing source STFT...")
    S_source, freqs = compute_stft(y_source, n_fft=n_fft, hop_length=hop_length)
    info['S_source'] = S_source
    
    # 3. Apply spectral modulation based on target pitch
    print("Applying spectral modulation...")
    S_modulated = spectral_modulation(
        S_source, f0_target, sr, n_fft,
        modulation_strength=alpha * 0.8,
        bandwidth_hz=80.0
    )
    info['S_modulated'] = S_modulated
    
    # 4. Reconstruct audio
    print("Reconstructing audio...")
    y_modulated = reconstruct_audio(S_modulated, hop_length=hop_length)
    
    # 5. Apply envelope transfer
    print("Transferring envelope...")
    y_output = envelope_transfer(
        y_modulated, env_target, env_times, sr,
        transfer_strength=alpha * 0.6,
        preserve_dynamics=True
    )
    
    # 6. Ensure same length as source
    if len(y_output) < len(y_source):
        y_output = np.pad(y_output, (0, len(y_source) - len(y_output)))
    else:
        y_output = y_output[:len(y_source)]
    
    # 7. Final blend with original source for smoothness
    y_output = alpha * y_output + (1 - alpha) * y_source[:len(y_output)]
    
    info['y_output'] = y_output
    
    return y_output, info

