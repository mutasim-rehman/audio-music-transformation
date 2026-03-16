"""
Neural Voice Cloning Core (PyTorch + DDSP architecture)

This module implements a simplified Differentiable Digital Signal Processing (DDSP)
autoencoder. It learns to synthesize a specific "voice" (e.g., clouds or wind)
driven by fundamental frequency (F0) and loudness contours.

Reference: DDSP: Differentiable Digital Signal Processing (Engel et al., 2020)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# SYNTHESIS COMPONENTS
# =====================================================================

class HarmonicOscillator(nn.Module):
    """
    Synthesizes audio from a combination of harmonic sine waves.
    Driven by fundamental frequency (F0) and harmonic amplitude distributions.
    """
    def __init__(self, sample_rate=22050):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, f0, harmonic_amplitudes):
        """
        f0: (batch, time, 1) - Hz
        harmonic_amplitudes: (batch, time, n_harmonics) - linear amplitude [0, 1]
        
        Returns:
            audio: (batch, time * hop_size)
        """
        # We will implement block-level synthesis for simplicity in this prototype.
        # A full production DDSP uses cumulative phase over the exact waveform length.
        # For simplicity in this PyTorch implementation, we will return parameters
        # that the standard DSP pipeline can synthesize, OR we process it via inverse STFT.
        
        # In a true DDSP model, this pushes waveforms. To keep the AI model lightweight
        # and not require custom CUDA kernels, we output the *spectral frames* and 
        # invert them with an ISTFT/Griffin-Lim or direct oscillator in inference.
        pass

class MLPDDSPEncoder(nn.Module):
    """
    A simple MLP that takes F0 and Loudness as input and predicts the control parameters
    for the synthesizer (harmonic distribution and noise filter coefficients).
    """
    def __init__(self, n_harmonics=60, n_noise_bands=65, hidden_size=256):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        
        # Input features: F0 (1 dim) + Loudness (1 dim) = 2
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
        )
        
        # Decoders
        self.out_harmonics = nn.Linear(hidden_size, n_harmonics)
        self.out_noise = nn.Linear(hidden_size, n_noise_bands)
        self.out_global_amp = nn.Linear(hidden_size, 1)

    def forward(self, f0_scaled, loudness_scaled):
        """
        Takes scaled F0 and Loudness contours and predicts matching spectra.
        f0_scaled: (batch, time, 1)
        loudness_scaled: (batch, time, 1)
        """
        x = torch.cat([f0_scaled, loudness_scaled], dim=-1)
        h = self.net(x)
        
        # Force positive outputs with modified sigmoid/softplus
        harmonics = F.softplus(self.out_harmonics(h))
        noise = F.softplus(self.out_noise(h))
        amp = torch.sigmoid(self.out_global_amp(h))
        
        # Normalize harmonic distribution to sum to 1
        harmonics = harmonics / (harmonics.sum(dim=-1, keepdim=True) + 1e-7)
        
        return amp, harmonics, noise

# =====================================================================
# DATASET & LOSS
# =====================================================================

class SpectralLoss(nn.Module):
    """Multi-Scale Spectral Loss (MSS Loss). Measures difference between true and generated spectrograms."""
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, y_true_mag, y_pred_mag):
        """
        Since we are outputting magnitude spectra directly in this prototype, 
        we use L1 and Log L1 on magnitudes directly.
        """
        # Linear loss
        lin_loss = F.l1_loss(y_true_mag, y_pred_mag)
        
        # Log loss
        log_loss = F.l1_loss(
            torch.log(y_true_mag + 1e-7), 
            torch.log(y_pred_mag + 1e-7)
        )
        return lin_loss + log_loss

# =====================================================================
# DATA EXTRACTION & SCALING
# =====================================================================

def extract_training_features(y, sr, hop_length=256, n_fft=2048):
    """
    Extracts True Spectrogram, F0, and Loudness from audio.
    Used to prepare the dataset.
    """
    import librosa
    from .analysis import extract_pitch, extract_envelope
    
    # Extract Spectrogram (Target for model to learn)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S).T # (time, freq)
    
    # Extract F0
    f0, voiced, _ = extract_pitch(y, sr, hop_length=hop_length)
    
    # Extract Loudness (A-weighted power is better, using envelope for now)
    loudness, _ = extract_envelope(y, sr, hop_length=hop_length, method='rms')
    
    # Ensure lengths match
    min_len = min(len(mag), len(f0), len(loudness))
    mag = mag[:min_len]
    f0 = f0[:min_len]
    loudness = loudness[:min_len]
    
    return mag, f0, loudness

def scale_features(f0, loudness):
    """Log scales for neural inputs"""
    # Hz to log Hz
    f0_scaled = np.where(f0 > 0, np.log2(f0 / 50.0), 0.0)
    
    # Loudness to dB-like scale
    loudness_scaled = np.log10(np.clip(loudness, 1e-5, None))
    # Normalize approx to [-1, 1] based on standard audio ranges
    loudness_scaled = (loudness_scaled + 5.0) / 5.0 
    
    return f0_scaled, loudness_scaled
