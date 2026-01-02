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
    
    # OUTPUT LENGTH = TARGET LENGTH (the song determines the duration!)
    n_samples_target = len(y_target)
    n_samples_source = len(y_source)
    
    # Extend or loop source to match target length if needed
    if n_samples_source < n_samples_target:
        repeats = int(np.ceil(n_samples_target / n_samples_source))
        y_source = np.tile(y_source, repeats)[:n_samples_target]
        print(f"Source looped {repeats}x to match target length ({n_samples_target/sr:.2f}s)")
    else:
        y_source = y_source[:n_samples_target]
    
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
    
    # 6. Ensure output is EXACTLY the target length
    if len(y_output) < n_samples_target:
        y_output = np.pad(y_output, (0, n_samples_target - len(y_output)))
    else:
        y_output = y_output[:n_samples_target]
    
    # 7. Final blend with original source for smoothness
    y_output = alpha * y_output + (1 - alpha) * y_source[:len(y_output)]
    
    info['y_output'] = y_output
    info['duration_seconds'] = n_samples_target / sr
    info['target_samples'] = n_samples_target
    
    # Final verification
    assert len(y_output) == n_samples_target, f"Length mismatch: {len(y_output)} vs {n_samples_target}"
    
    minutes = int(n_samples_target / sr // 60)
    seconds = (n_samples_target / sr) % 60
    print(f"Output duration: {minutes}:{seconds:05.2f} (matches target exactly)")
    
    return y_output, info


def vocoder_singing_texture(
    y_source: np.ndarray,
    y_target: np.ndarray,
    sr: int = 22050,
    alpha: float = 0.7,
    n_harmonics: int = 12,
    texture_amount: float = 0.4,
    formant_shift: float = 1.0,
    hop_length: int = 256
) -> Tuple[np.ndarray, dict]:
    """
    🎵 VOCODER-STYLE SINGING TEXTURE - Makes ambient sounds truly "sing"!
    
    This is a more powerful approach that GENERATES harmonic tones from the 
    ambient sound, creating the effect of the sound source "singing" the melody.
    
    THE CORE IDEA:
    When you imagine clouds singing, your brain:
    1. Takes the pitch contour of a song (the melody)
    2. Shapes the ambient sound to follow that pitch
    3. Creates the illusion that the clouds ARE the singer
    
    This function does exactly that by:
    1. Extracting melody (pitch) and rhythm (envelope) from the target song
    2. Generating harmonics at those pitches
    3. Using the source sound's texture to "color" those harmonics
    4. The result: ambient sound that SINGS the song!
    
    Parameters
    ----------
    y_source : np.ndarray
        Source audio (ambient sound - clouds, wind, fan, rain, etc.)
    y_target : np.ndarray
        Target audio (the song - vocals with melody you want the clouds to sing)
    sr : int
        Sample rate (default 22050)
    alpha : float [0, 1]
        How much "singing" vs original texture
        0.0 = pure source (just clouds)
        1.0 = maximum singing effect
        0.6-0.8 = usually best balance
    n_harmonics : int
        Number of harmonics to synthesize (more = richer tone)
        8-15 is good for voice-like quality
    texture_amount : float [0, 1]
        How much of the original source texture to preserve
        Higher = more "cloudy", Lower = more pure tone
    formant_shift : float
        Shift formants (vocal resonances) up/down
        1.0 = neutral, >1 = brighter/smaller voice, <1 = darker/larger
    hop_length : int
        Analysis hop size (smaller = better time resolution)
    
    Returns
    -------
    y_output : np.ndarray
        The magical output - your ambient sound singing the song!
    info : dict
        Intermediate data for visualization
    
    LEARNING - HOW IT WORKS:
    ========================
    
    Step 1: ANALYZE THE SONG
    - Extract F0 (pitch) - this IS the melody
    - Extract envelope - this IS the rhythm/phrasing
    - Extract formants - these make vowels sound like vowels
    
    Step 2: BUILD A HARMONIC SYNTHESIZER
    - For each time frame, generate sine waves at F0, 2*F0, 3*F0...
    - These harmonics create a pitched tone from "nothing"
    
    Step 3: TEXTURE IT WITH THE SOURCE
    - Measure how much energy the source (clouds) has at each frequency
    - Use this to modulate the harmonics' amplitudes
    - Result: harmonics that "sound like" the source material
    
    Step 4: SHAPE WITH ENVELOPE
    - Apply the song's dynamics (loud/soft patterns)
    - This gives it natural phrasing and rhythm
    
    Step 5: ADD FORMANTS (OPTIONAL)
    - Voice has resonances that make "ah", "ee", "oo" sounds
    - Adding these makes it sound more vocal/sung
    
    The math: y_out = α * Synthesized_Harmonics * Source_Texture * Target_Envelope
                    + (1-α) * Filtered_Source
    """
    from .analysis import compute_stft, extract_pitch, extract_envelope
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt
    
    info = {}
    
    # OUTPUT LENGTH = TARGET LENGTH (the song determines the duration!)
    n_samples_target = len(y_target)
    n_samples_source = len(y_source)
    
    # Calculate target duration for display
    target_minutes = int(n_samples_target / sr // 60)
    target_seconds = (n_samples_target / sr) % 60
    print(f"[VOCODER] Target song duration: {target_minutes}:{target_seconds:05.2f}")
    print(f"[VOCODER] Output will be EXACTLY {target_minutes}:{target_seconds:05.2f}")
    
    # Extend or loop source to match target length if needed
    if n_samples_source < n_samples_target:
        # Loop the source audio to match target length
        repeats = int(np.ceil(n_samples_target / n_samples_source))
        y_source = np.tile(y_source, repeats)[:n_samples_target]
        print(f"[VOCODER] Source looped {repeats}x to match target")
    else:
        # Trim source to target length
        y_source = y_source[:n_samples_target]
    
    n_samples = n_samples_target  # Output will be exactly as long as target
    
    # ========== STEP 1: ANALYZE TARGET (THE SONG) ==========
    print("[VOCODER] Analyzing target song...")
    
    # Extract pitch contour (THE MELODY!)
    print("  -> Extracting pitch (melody)...")
    f0_target, voiced_target, times_pitch = extract_pitch(
        y_target, sr, hop_length=hop_length, fmin=50, fmax=800
    )
    info['f0'] = f0_target
    info['voiced'] = voiced_target
    info['times'] = times_pitch
    
    # Extract envelope (THE RHYTHM!)
    print("  -> Extracting envelope (dynamics)...")
    env_target, times_env = extract_envelope(y_target, sr, hop_length=hop_length)
    info['envelope'] = env_target
    
    # Smooth the pitch contour for more natural transitions
    f0_smoothed = smooth_pitch_contour(f0_target, voiced_target, smoothing=0.8)
    
    # ========== STEP 2: ANALYZE SOURCE TEXTURE ==========
    print("[VOCODER] Analyzing source texture...")
    
    # Get spectral characteristics of the source
    S_source, _ = compute_stft(y_source, n_fft=2048, hop_length=hop_length)
    source_magnitude = np.abs(S_source)
    
    # Compute average spectral shape of source (its "color")
    source_spectral_shape = np.mean(source_magnitude, axis=1)
    source_spectral_shape = source_spectral_shape / (np.max(source_spectral_shape) + 1e-10)
    info['source_spectrum'] = source_spectral_shape
    
    # ========== STEP 3: SYNTHESIZE HARMONICS ==========
    print("[VOCODER] Synthesizing harmonics at target pitches...")
    
    # Create time array at sample rate
    t = np.arange(n_samples) / sr
    
    # Interpolate pitch to sample rate
    if len(f0_smoothed) > 1:
        pitch_interp = interp1d(
            times_pitch, f0_smoothed, 
            kind='linear', fill_value=0, bounds_error=False
        )
        f0_at_samples = pitch_interp(t)
    else:
        f0_at_samples = np.zeros(n_samples)
    
    # Interpolate voiced flag
    voiced_interp = interp1d(
        times_pitch, voiced_target.astype(float),
        kind='nearest', fill_value=0, bounds_error=False
    )
    voiced_at_samples = voiced_interp(t) > 0.5
    
    # Generate harmonics using additive synthesis
    y_harmonics = np.zeros(n_samples)
    
    # Phase accumulator for smooth pitch transitions (no clicks!)
    phase = np.zeros(n_harmonics)
    
    # Harmonic amplitude weights (natural voice rolloff)
    harmonic_weights = np.array([1.0 / (h ** 0.7) for h in range(1, n_harmonics + 1)])
    harmonic_weights = harmonic_weights / np.sum(harmonic_weights)
    
    # Pre-compute frequency bins for texture mapping
    freq_bins = np.linspace(0, sr/2, len(source_spectral_shape))
    
    # Block-based synthesis for efficiency
    block_size = hop_length
    n_blocks = n_samples // block_size
    
    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, n_samples)
        block_t = t[start:end]
        
        # Get pitch for this block
        f0_block = np.mean(f0_at_samples[start:end])
        is_voiced = np.mean(voiced_at_samples[start:end]) > 0.5
        
        if f0_block > 0 and is_voiced:
            # Generate each harmonic
            for h in range(n_harmonics):
                harmonic_freq = f0_block * (h + 1)
                
                if harmonic_freq < sr / 2:  # Nyquist limit
                    # Get texture weight from source spectrum
                    freq_idx = np.argmin(np.abs(freq_bins - harmonic_freq))
                    texture_weight = source_spectral_shape[min(freq_idx, len(source_spectral_shape)-1)]
                    texture_weight = texture_amount + (1 - texture_amount) * texture_weight
                    
                    # Phase accumulation for smooth transitions
                    phase_increment = 2 * np.pi * harmonic_freq / sr
                    phases = phase[h] + np.cumsum(np.full(end - start, phase_increment))
                    phase[h] = phases[-1] % (2 * np.pi)
                    
                    # Add this harmonic
                    amplitude = harmonic_weights[h] * texture_weight
                    y_harmonics[start:end] += amplitude * np.sin(phases)
    
    # Normalize harmonics
    if np.max(np.abs(y_harmonics)) > 0:
        y_harmonics = y_harmonics / np.max(np.abs(y_harmonics))
    
    info['harmonics_raw'] = y_harmonics.copy()
    
    # ========== STEP 4: APPLY ENVELOPE ==========
    print("[VOCODER] Applying rhythm envelope...")
    
    # Interpolate envelope to sample rate
    env_interp = interp1d(
        times_env, env_target,
        kind='linear', fill_value=0, bounds_error=False
    )
    envelope_at_samples = env_interp(t)
    envelope_at_samples = envelope_at_samples / (np.max(envelope_at_samples) + 1e-10)
    
    # Apply envelope with slight attack smoothing
    envelope_smoothed = smooth_envelope(envelope_at_samples, sr, attack_ms=10, release_ms=50)
    
    y_harmonics_shaped = y_harmonics * envelope_smoothed
    info['harmonics_shaped'] = y_harmonics_shaped.copy()
    
    # ========== STEP 5: ADD FORMANT RESONANCES ==========
    print("[VOCODER] Adding vocal formants...")
    
    # Typical vowel formants (blend of several vowels for natural sound)
    formants = np.array([500, 1500, 2500, 3500]) * formant_shift
    formant_bandwidths = np.array([100, 120, 150, 200])
    formant_gains = np.array([1.0, 0.7, 0.5, 0.3])
    
    y_formant = apply_formant_filter(
        y_harmonics_shaped, sr, formants, formant_bandwidths, formant_gains
    )
    
    # ========== STEP 6: BLEND WITH TEXTURED SOURCE ==========
    print("[VOCODER] Blending with source texture...")
    
    # Filter source to match pitch regions (bandpass around melody)
    y_source_filtered = bandpass_follow_pitch(
        y_source, f0_at_samples, sr, bandwidth_ratio=3.0
    )
    
    # Apply same envelope to filtered source
    y_source_shaped = y_source_filtered * envelope_smoothed * 0.5
    
    # Final mix
    y_singing = alpha * y_formant + (1 - alpha * 0.5) * y_source_shaped
    
    # Normalize
    y_output = y_singing / (np.max(np.abs(y_singing)) + 1e-10) * 0.9
    
    # Ensure output is EXACTLY the target length
    if len(y_output) != n_samples_target:
        if len(y_output) < n_samples_target:
            y_output = np.pad(y_output, (0, n_samples_target - len(y_output)))
        else:
            y_output = y_output[:n_samples_target]
    
    info['y_output'] = y_output
    info['duration_seconds'] = n_samples_target / sr
    info['target_samples'] = n_samples_target
    
    # Final verification
    assert len(y_output) == n_samples_target, f"Length mismatch: {len(y_output)} vs {n_samples_target}"
    
    minutes = int(n_samples_target / sr // 60)
    seconds = (n_samples_target / sr) % 60
    print(f"[VOCODER] Done! Output duration: {minutes}:{seconds:05.2f} (matches target exactly)")
    
    return y_output, info


def smooth_pitch_contour(f0: np.ndarray, voiced: np.ndarray, smoothing: float = 0.5) -> np.ndarray:
    """Smooth pitch contour for natural transitions between notes."""
    f0_smooth = f0.copy()
    
    # Fill unvoiced gaps with interpolation
    voiced_indices = np.where(voiced)[0]
    if len(voiced_indices) > 1:
        from scipy.interpolate import interp1d
        
        # Only interpolate, don't extrapolate
        interp_func = interp1d(
            voiced_indices, f0[voiced_indices],
            kind='linear', fill_value=0, bounds_error=False
        )
        
        # Fill gaps
        unvoiced_indices = np.where(~voiced)[0]
        for idx in unvoiced_indices:
            if voiced_indices[0] < idx < voiced_indices[-1]:
                f0_smooth[idx] = interp_func(idx)
    
    # Apply smoothing filter
    if smoothing > 0:
        from scipy.ndimage import uniform_filter1d
        window = max(1, int(len(f0_smooth) * smoothing * 0.05))
        f0_smooth = uniform_filter1d(f0_smooth, size=window)
    
    return f0_smooth


def smooth_envelope(envelope: np.ndarray, sr: int, attack_ms: float = 10, release_ms: float = 50) -> np.ndarray:
    """Apply attack/release smoothing to envelope."""
    from scipy.signal import lfilter
    
    attack_samples = int(sr * attack_ms / 1000)
    release_samples = int(sr * release_ms / 1000)
    
    # Simple one-pole smoothing
    smoothed = np.zeros_like(envelope)
    attack_coef = 1.0 - np.exp(-1.0 / max(1, attack_samples))
    release_coef = 1.0 - np.exp(-1.0 / max(1, release_samples))
    
    current = 0.0
    for i, target in enumerate(envelope):
        if target > current:
            current += attack_coef * (target - current)
        else:
            current += release_coef * (target - current)
        smoothed[i] = current
    
    return smoothed


def apply_formant_filter(y: np.ndarray, sr: int, formants: np.ndarray, 
                         bandwidths: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply formant resonances to make sound more vocal."""
    from scipy.signal import iirpeak, lfilter
    
    y_out = np.zeros_like(y)
    
    for f, bw, g in zip(formants, bandwidths, gains):
        if f < sr / 2:  # Below Nyquist
            # Create resonant peak filter
            Q = f / bw
            b, a = iirpeak(f / (sr / 2), Q)
            
            # Apply filter
            y_filtered = lfilter(b, a, y)
            y_out += g * y_filtered
    
    # Normalize
    if np.max(np.abs(y_out)) > 0:
        y_out = y_out / np.max(np.abs(y_out))
    
    return y_out


def puppet_clouds_with_voice(
    y_clouds: np.ndarray,
    y_voice: np.ndarray,
    sr: int = 22050,
    influence: float = 0.7,
    hop_length: int = 256,
    n_fft: int = 2048
) -> Tuple[np.ndarray, dict]:
    """
    PUPPETEERING APPROACH: Force clouds to move and resonate like a singer.
    
    PHILOSOPHY:
    -----------
    You don't turn clouds into Elvis - you force clouds to move the way Elvis does.
    
    - Clouds = the BODY (material, texture, breath)
    - Voice = the MOTION (pitch movement, timing, vocal shape)
    
    This is NOT cloning. This is PUPPETEERING.
    Clouds are the puppet body. Elvis is the strings. You are the director.
    
    THE LOGIC:
    ----------
    1. Strip clouds to essence (texture, loudness, temporal flow)
    2. Extract voice BEHAVIOR (pitch trajectory, rhythm, vocal shape)
    3. Force clouds to follow that behavior
    4. Bend the noise so its frequencies move with the melody
    5. Blend, don't overwrite
    
    Parameters
    ----------
    y_clouds : np.ndarray
        The cloud/ambient sound - this is your RAW MATERIAL
    y_voice : np.ndarray  
        The voice/song - this provides the BEHAVIOR to copy
    sr : int
        Sample rate
    influence : float [0, 1]
        How much the voice controls the clouds:
        Low (0.3) = whispering clouds
        Medium (0.5) = cloud choir  
        High (0.8) = sky singing
    hop_length : int
        Analysis frame hop
    n_fft : int
        FFT size
    
    Returns
    -------
    y_output : np.ndarray
        Clouds that move like the singer (same length as voice)
    info : dict
        Intermediate data
    
    ONE-LINE SUMMARY:
    You don't teach clouds to sing. You teach them how to move their breath like a singer.
    """
    import librosa
    from .analysis import compute_stft, extract_pitch, extract_envelope
    from .synthesis import reconstruct_audio
    from scipy.interpolate import interp1d
    from scipy.signal import medfilt
    
    info = {}
    
    # Output length = voice length (the song determines duration)
    n_samples_voice = len(y_voice)
    n_samples_clouds = len(y_clouds)
    
    target_duration = n_samples_voice / sr
    print(f"[PUPPET] Voice duration: {int(target_duration//60)}:{target_duration%60:05.2f}")
    print(f"[PUPPET] Output will match voice exactly")
    
    # Loop clouds to match voice length
    if n_samples_clouds < n_samples_voice:
        repeats = int(np.ceil(n_samples_voice / n_samples_clouds))
        y_clouds = np.tile(y_clouds, repeats)[:n_samples_voice]
        print(f"[PUPPET] Clouds looped {repeats}x to match voice")
    else:
        y_clouds = y_clouds[:n_samples_voice]
    
    # ========== STEP 1: STRIP CLOUDS TO ESSENCE ==========
    print("[PUPPET] Step 1: Stripping clouds to essence...")
    
    # Compute STFT of clouds
    S_clouds = librosa.stft(y_clouds, n_fft=n_fft, hop_length=hop_length)
    mag_clouds = np.abs(S_clouds)
    phase_clouds = np.angle(S_clouds)
    
    # Extract cloud's spectral envelope (its "texture shape")
    # This is what clouds are "made of"
    cloud_spectral_envelope = np.mean(mag_clouds, axis=1)
    cloud_spectral_envelope = cloud_spectral_envelope / (np.max(cloud_spectral_envelope) + 1e-10)
    info['cloud_essence'] = cloud_spectral_envelope
    
    # Extract cloud's temporal envelope (how loud over time)
    cloud_envelope, cloud_env_times = extract_envelope(y_clouds, sr, hop_length=hop_length)
    info['cloud_envelope'] = cloud_envelope
    
    # ========== STEP 2: LISTEN TO VOICE - EXTRACT BEHAVIOR ==========
    print("[PUPPET] Step 2: Extracting voice behavior (motion, not sound)...")
    
    # Extract pitch trajectory (HOW notes rise and fall)
    f0_voice, voiced_voice, times_voice = extract_pitch(
        y_voice, sr, hop_length=hop_length, fmin=50, fmax=800
    )
    info['pitch_trajectory'] = f0_voice
    info['voiced_mask'] = voiced_voice
    info['times'] = times_voice
    
    # Extract rhythm/timing (energy envelope)
    voice_envelope, voice_env_times = extract_envelope(y_voice, sr, hop_length=hop_length)
    info['voice_rhythm'] = voice_envelope
    
    # Extract vocal shape (formants - mouth position encoded in frequencies)
    S_voice = librosa.stft(y_voice, n_fft=n_fft, hop_length=hop_length)
    mag_voice = np.abs(S_voice)
    
    # Compute spectral envelope of voice (the "vocal tract shape")
    # This tells us WHERE the voice puts its energy
    voice_spectral_shape = np.zeros_like(mag_voice)
    for t in range(mag_voice.shape[1]):
        # Smooth spectral envelope - this is the mouth/throat shape
        frame = mag_voice[:, t]
        if np.max(frame) > 0:
            # Use cepstral smoothing to get spectral envelope
            frame_smooth = np.maximum(medfilt(frame, kernel_size=31), 1e-10)
            voice_spectral_shape[:, t] = frame_smooth / np.max(frame_smooth)
    info['vocal_shape'] = voice_spectral_shape
    
    # ========== STEP 3: FORCE CLOUDS TO FOLLOW VOICE MOTION ==========
    print("[PUPPET] Step 3: Forcing clouds to follow voice motion...")
    
    n_frames = min(mag_clouds.shape[1], mag_voice.shape[1])
    n_freqs = mag_clouds.shape[0]
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Prepare output magnitude
    mag_output = np.zeros((n_freqs, n_frames))
    
    # Smooth pitch for natural movement
    f0_smooth = f0_voice.copy()
    voiced_indices = np.where(voiced_voice)[0]
    if len(voiced_indices) > 1:
        # Interpolate through unvoiced regions
        interp_func = interp1d(
            voiced_indices, f0_voice[voiced_indices],
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
        f0_smooth = interp_func(np.arange(len(f0_voice)))
        f0_smooth = np.maximum(f0_smooth, 0)
    
    # Resample to match frames
    if len(f0_smooth) != n_frames:
        f0_smooth = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(f0_smooth)),
            f0_smooth
        )
    if len(voiced_voice) != n_frames:
        voiced_resampled = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(voiced_voice)),
            voiced_voice.astype(float)
        ) > 0.5
    else:
        voiced_resampled = voiced_voice[:n_frames]
    
    # For each frame, bend the cloud's frequencies to follow the voice
    for t in range(n_frames):
        cloud_frame = mag_clouds[:, t]
        f0 = f0_smooth[t] if t < len(f0_smooth) else 0
        is_voiced = voiced_resampled[t] if t < len(voiced_resampled) else False
        
        if f0 > 50 and is_voiced:
            # === BEND THE NOISE ===
            # Create a modulation that pulls cloud energy toward harmonic frequencies
            
            # Build harmonic attractor (where we want the energy to go)
            harmonic_attractor = np.zeros(n_freqs)
            n_harmonics = 12
            
            for h in range(1, n_harmonics + 1):
                harmonic_freq = f0 * h
                if harmonic_freq < sr / 2:
                    # Find closest frequency bin
                    bin_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                    
                    # Create attraction zone with bandwidth
                    bandwidth_bins = max(5, int(n_freqs * 0.02 / h))  # Narrower for higher harmonics
                    for offset in range(-bandwidth_bins, bandwidth_bins + 1):
                        idx = bin_idx + offset
                        if 0 <= idx < n_freqs:
                            # Gaussian attraction
                            weight = np.exp(-0.5 * (offset / (bandwidth_bins/2))**2)
                            strength = 1.0 / (h ** 0.5)  # Higher harmonics weaker
                            harmonic_attractor[idx] = max(harmonic_attractor[idx], weight * strength)
            
            # Normalize attractor
            if np.max(harmonic_attractor) > 0:
                harmonic_attractor = harmonic_attractor / np.max(harmonic_attractor)
            
            # === APPLY VOCAL SHAPE ===
            # The voice's spectral envelope tells us mouth/throat position
            if t < voice_spectral_shape.shape[1]:
                vocal_shape = voice_spectral_shape[:, t]
            else:
                vocal_shape = np.ones(n_freqs)
            
            # === COMBINE: Cloud material + Voice behavior ===
            # The cloud provides the raw energy
            # The harmonic attractor pulls it toward pitch
            # The vocal shape sculpts the overall timbre
            
            modulation = (
                (1 - influence) * np.ones(n_freqs) +  # Keep some original
                influence * harmonic_attractor * vocal_shape  # Add voice behavior
            )
            
            # Apply modulation to cloud material
            mag_output[:, t] = cloud_frame * modulation
            
        else:
            # Unvoiced: clouds breathe quietly (reduced, but not silent)
            mag_output[:, t] = cloud_frame * (1 - influence * 0.7)
    
    info['mag_modulated'] = mag_output
    
    # ========== STEP 4: APPLY RHYTHM (VOICE'S BREATHING) ==========
    print("[PUPPET] Step 4: Applying voice rhythm (breathing pattern)...")
    
    # Resample voice envelope to match frames
    if len(voice_envelope) != n_frames:
        voice_env_resampled = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(voice_envelope)),
            voice_envelope
        )
    else:
        voice_env_resampled = voice_envelope[:n_frames]
    
    # Normalize envelope
    voice_env_resampled = voice_env_resampled / (np.max(voice_env_resampled) + 1e-10)
    
    # Blend cloud's natural rhythm with voice's rhythm
    if len(cloud_envelope) != n_frames:
        cloud_env_resampled = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(cloud_envelope)),
            cloud_envelope
        )
    else:
        cloud_env_resampled = cloud_envelope[:n_frames]
    cloud_env_resampled = cloud_env_resampled / (np.max(cloud_env_resampled) + 1e-10)
    
    # Combined rhythm: voice leads, cloud follows
    combined_envelope = (
        influence * voice_env_resampled + 
        (1 - influence) * cloud_env_resampled
    )
    
    # Apply rhythm to magnitude
    for t in range(n_frames):
        mag_output[:, t] = mag_output[:, t] * combined_envelope[t]
    
    info['rhythm_envelope'] = combined_envelope
    
    # ========== STEP 5: RECONSTRUCT - CLOUDS THAT MOVE LIKE SINGER ==========
    print("[PUPPET] Step 5: Reconstructing puppeteered clouds...")
    
    # Use original cloud phase (keeps the "air" quality)
    phase_output = phase_clouds[:, :n_frames]
    
    # Reconstruct complex STFT
    S_output = mag_output * np.exp(1j * phase_output)
    
    # Inverse STFT
    y_output = librosa.istft(S_output, hop_length=hop_length, length=n_samples_voice)
    
    # Ensure exact length match
    if len(y_output) < n_samples_voice:
        y_output = np.pad(y_output, (0, n_samples_voice - len(y_output)))
    else:
        y_output = y_output[:n_samples_voice]
    
    # Normalize
    y_output = y_output / (np.max(np.abs(y_output)) + 1e-10) * 0.9
    
    info['y_output'] = y_output
    info['duration_seconds'] = n_samples_voice / sr
    
    output_duration = n_samples_voice / sr
    print(f"[PUPPET] Done! Output: {int(output_duration//60)}:{output_duration%60:05.2f} (matches voice)")
    print(f"[PUPPET] Clouds now move like the singer!")
    
    return y_output, info


def bandpass_follow_pitch(y: np.ndarray, f0: np.ndarray, sr: int, 
                          bandwidth_ratio: float = 2.0) -> np.ndarray:
    """Bandpass filter source to follow pitch contour."""
    from scipy.signal import butter, filtfilt
    
    y_out = np.zeros_like(y)
    hop = len(y) // len(f0)
    
    for i in range(len(f0)):
        start = i * hop
        end = min(start + hop, len(y))
        
        if f0[i] > 50:  # Only filter if there's a valid pitch
            # Bandpass around pitch
            low = max(30, f0[i] / bandwidth_ratio) / (sr / 2)
            high = min(0.99, f0[i] * bandwidth_ratio / (sr / 2))
            
            if low < high:
                try:
                    b, a = butter(2, [low, high], btype='band')
                    y_out[start:end] = filtfilt(b, a, y[start:end], padlen=min(10, len(y[start:end])-1))
                except:
                    y_out[start:end] = y[start:end]
            else:
                y_out[start:end] = y[start:end]
        else:
            y_out[start:end] = y[start:end] * 0.3  # Reduce unvoiced regions
    
    return y_out