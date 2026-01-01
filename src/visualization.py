"""
Visualization Module

Tools for visualizing audio signals and transformations.

LEARNING NOTES:
- Good visualizations help understand what's happening to the audio
- Spectrograms show frequency content over time
- Waveforms show amplitude over time
- Comparing before/after reveals transformation effects
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple, List
from matplotlib.figure import Figure


# Set a beautiful default style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.family'] = 'sans-serif'


def plot_waveform(
    y: np.ndarray,
    sr: int = 22050,
    title: str = "Waveform",
    ax: Optional[plt.Axes] = None,
    color: str = '#e94560'
) -> plt.Axes:
    """
    Plot audio waveform.
    
    LEARNING:
    ---------
    The waveform shows amplitude (loudness) over time.
    - Tall peaks = loud moments
    - Dense oscillations = high frequency content
    - Sparse waves = low frequency content
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color=color, linewidth=0.5, alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color=color)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, times[-1]])
    
    return ax


def plot_spectrogram(
    S: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    title: str = "Spectrogram",
    ax: Optional[plt.Axes] = None,
    y_axis: str = 'log',
    cmap: str = 'magma'
) -> plt.Axes:
    """
    Plot spectrogram (magnitude of STFT).
    
    Parameters
    ----------
    S : np.ndarray
        STFT matrix (complex or magnitude)
    sr : int
        Sample rate
    hop_length : int
        Hop size used in STFT
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on
    y_axis : str
        'log' (logarithmic) or 'linear'
    cmap : str
        Colormap name
    
    LEARNING:
    ---------
    The spectrogram is a 2D image where:
    - X-axis = Time
    - Y-axis = Frequency
    - Color = Magnitude (energy)
    
    Bright horizontal lines = sustained tones
    Vertical patterns = transients/attacks
    Diffuse patterns = noise-like sounds
    
    Log frequency scale (y_axis='log') matches human perception better.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get magnitude if complex
    if np.iscomplexobj(S):
        S = np.abs(S)
    
    # Convert to dB for better visualization
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis,
        ax=ax,
        cmap=cmap
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Magnitude (dB)')
    
    return ax


def plot_mel_spectrogram(
    S_mel: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'inferno'
) -> plt.Axes:
    """
    Plot mel-scaled spectrogram.
    
    LEARNING:
    ---------
    Mel spectrograms compress frequency information to match
    human perception. Lower frequencies have more resolution
    than higher frequencies.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    S_db = librosa.power_to_db(S_mel, ref=np.max)
    
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap=cmap
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Magnitude (dB)')
    
    return ax


def plot_pitch_contour(
    f0: np.ndarray,
    times: np.ndarray,
    voiced_flag: Optional[np.ndarray] = None,
    title: str = "Pitch Contour",
    ax: Optional[plt.Axes] = None,
    color: str = '#00ff88'
) -> plt.Axes:
    """
    Plot pitch (F0) contour over time.
    
    LEARNING:
    ---------
    The pitch contour is the melody!
    - Rising line = pitch going up
    - Falling line = pitch going down
    - Gaps = unvoiced regions (breaths, consonants)
    
    This is what we extract from Elvis and impose on the clouds.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Mask unvoiced regions
    f0_plot = f0.copy()
    f0_plot[f0_plot == 0] = np.nan
    
    ax.plot(times, f0_plot, color=color, linewidth=2, marker='o', 
            markersize=2, alpha=0.8, label='F0')
    
    if voiced_flag is not None:
        # Highlight voiced regions
        ax.fill_between(times, 0, np.max(f0_plot[~np.isnan(f0_plot)]),
                       where=voiced_flag, alpha=0.1, color=color)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([0, None])
    
    return ax


def plot_envelope(
    envelope: np.ndarray,
    times: np.ndarray,
    title: str = "Amplitude Envelope",
    ax: Optional[plt.Axes] = None,
    color: str = '#ff6b6b'
) -> plt.Axes:
    """
    Plot amplitude envelope.
    
    LEARNING:
    ---------
    The envelope shows the dynamics - how loud/soft the sound is over time.
    This captures the "breathing" and "phrasing" of a performance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    
    ax.plot(times, envelope, color=color, linewidth=2)
    ax.fill_between(times, 0, envelope, alpha=0.3, color=color)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([0, None])
    
    return ax


def plot_comparison(
    y_source: np.ndarray,
    y_target: np.ndarray,
    y_output: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Figure:
    """
    Create comprehensive comparison plot of source, target, and output.
    
    LEARNING:
    ---------
    This visualization helps you see how the transformation worked:
    - Top row: Waveforms (amplitude over time)
    - Bottom row: Spectrograms (frequency over time)
    
    Compare the output to both source and target to see what was preserved
    and what was transferred.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Audio Transformation Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    # Waveforms
    plot_waveform(y_source, sr, "Source (Environmental)", axes[0, 0], '#4ecdc4')
    plot_waveform(y_target, sr, "Target (Vocal)", axes[0, 1], '#ff6b6b')
    plot_waveform(y_output, sr, "Output (Transformed)", axes[0, 2], '#ffe66d')
    
    # Spectrograms
    from .analysis import compute_stft
    S_source, _ = compute_stft(y_source, hop_length=hop_length)
    S_target, _ = compute_stft(y_target, hop_length=hop_length)
    S_output, _ = compute_stft(y_output, hop_length=hop_length)
    
    plot_spectrogram(S_source, sr, hop_length, "Source Spectrogram", axes[1, 0], cmap='viridis')
    plot_spectrogram(S_target, sr, hop_length, "Target Spectrogram", axes[1, 1], cmap='plasma')
    plot_spectrogram(S_output, sr, hop_length, "Output Spectrogram", axes[1, 2], cmap='magma')
    
    plt.tight_layout()
    return fig


def plot_svd_components(
    U: np.ndarray,
    s: np.ndarray,
    n_show: int = 5,
    sr: int = 22050,
    n_fft: int = 2048
) -> Figure:
    """
    Visualize SVD components of a spectrogram.
    
    LEARNING:
    ---------
    SVD decomposes the spectrogram into components:
    - U columns: Spectral templates (frequency patterns)
    - s values: How important each template is
    
    The first few components often capture the main character of the sound.
    """
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3 * (n_show + 1), 4))
    
    # Plot singular values
    ax = axes[0]
    ax.bar(range(len(s)), s, color='#e94560')
    ax.set_xlabel('Component')
    ax.set_ylabel('Singular Value')
    ax.set_title('Component Importance', fontweight='bold')
    
    # Plot top components
    frequencies = np.linspace(0, sr / 2, U.shape[0])
    
    for i in range(min(n_show, U.shape[1])):
        ax = axes[i + 1]
        ax.plot(frequencies, U[:, i], color=plt.cm.viridis(i / n_show))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Weight')
        ax.set_title(f'Component {i+1}', fontweight='bold')
        ax.set_xlim([0, 4000])  # Focus on audible range
    
    plt.tight_layout()
    return fig


def plot_transformation_pipeline(info: dict, sr: int = 22050, hop_length: int = 512) -> Figure:
    """
    Visualize the full transformation pipeline.
    
    Shows each step: pitch extraction, modulation, envelope transfer.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Transformation Pipeline', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Target pitch
    if 'f0_target' in info:
        times = np.arange(len(info['f0_target'])) * hop_length / sr
        ax = axes[0, 0]
        f0 = info['f0_target'].copy()
        f0[f0 == 0] = np.nan
        ax.plot(times, f0, color='#00ff88', linewidth=2)
        ax.set_title('1. Target Pitch Contour', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    
    # 2. Target envelope
    if 'envelope_target' in info:
        times = np.arange(len(info['envelope_target'])) * hop_length / sr
        ax = axes[0, 1]
        ax.plot(times, info['envelope_target'], color='#ff6b6b', linewidth=2)
        ax.fill_between(times, 0, info['envelope_target'], alpha=0.3, color='#ff6b6b')
        ax.set_title('2. Target Amplitude Envelope', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    # 3. Source spectrogram
    if 'S_source' in info:
        plot_spectrogram(info['S_source'], sr, hop_length, 
                        '3. Source Spectrogram', axes[1, 0], cmap='viridis')
    
    # 4. Modulated spectrogram
    if 'S_modulated' in info:
        plot_spectrogram(info['S_modulated'], sr, hop_length,
                        '4. After Pitch Modulation', axes[1, 1], cmap='plasma')
    
    # 5. Modulation difference
    if 'S_source' in info and 'S_modulated' in info:
        diff = np.abs(info['S_modulated']) - np.abs(info['S_source'])
        ax = axes[2, 0]
        img = ax.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax.set_title('5. Modulation Effect (difference)', fontweight='bold')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Frequency Bin')
        plt.colorbar(img, ax=ax)
    
    # 6. Final output waveform
    if 'y_output' in info:
        plot_waveform(info['y_output'], sr, '6. Final Output', axes[2, 1], '#ffe66d')
    
    plt.tight_layout()
    return fig


def create_interactive_player(
    audio_dict: dict,
    sr: int = 22050
) -> None:
    """
    Create interactive audio players for comparing multiple audio signals.
    
    For use in Jupyter notebooks.
    
    LEARNING:
    ---------
    Listening is the ultimate test! Compare:
    - Does the output retain the source texture?
    - Can you hear the target melody/rhythm?
    - Is the blend natural or artificial?
    """
    try:
        from IPython.display import display, Audio, HTML
        
        display(HTML("<h3>🎧 Audio Comparison</h3>"))
        
        for name, y in audio_dict.items():
            display(HTML(f"<b>{name}</b>"))
            display(Audio(y, rate=sr))
            
    except ImportError:
        print("Interactive player requires IPython (Jupyter notebook)")
        print("Available audio signals:", list(audio_dict.keys()))

