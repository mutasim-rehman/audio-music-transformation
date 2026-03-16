"""
Train a Voice Model (DDSP-style) on a base ambient sound.

Usage:
    python train_voice.py --base samples/source_clouds.wav --output models/voice_clouds.pt --epochs 100
"""

import argparse
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.audio_io import load_audio
from src.neural_core import MLPDDSPEncoder, SpectralLoss, extract_training_features, scale_features

def prepare_dataset(audio_path, sr=22050, hop_length=256, n_fft=2048, batch_size=32, seq_len=100):
    """
    Slices a continuous audio file into (Batch, Time, Features) chunks for PyTorch.
    """
    print(f"Loading '{audio_path}'...")
    y, _ = load_audio(audio_path, sr=sr)
    
    print("Extracting acoustic features (F0, Loudness, Spectrogram)...")
    import librosa
    # Use our simplified DDSP feature extractor block
    from src.analysis import extract_pitch, extract_envelope
    
    # Target Spectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S).T  # (Total_Frames, Freq_Bins)
    
    # Input F0 + Loudness
    f0, voiced, _ = extract_pitch(y, sr, hop_length=hop_length)
    loudness, _ = extract_envelope(y, sr, hop_length=hop_length)
    
    # Make same length
    min_frames = min(len(mag), len(f0), len(loudness))
    mag = mag[:min_frames]
    f0 = f0[:min_frames]
    loudness = loudness[:min_frames]
    
    # Scale inputs
    f0_scaled, loud_scaled = scale_features(f0, loudness)
    
    # Slice into sequences of length 'seq_len'
    n_seqs = min_frames // seq_len
    
    print(f"Creating dataset with {n_seqs} sequences (Length: {seq_len} frames)")
    
    # Shape: (N, Seq, Freq)
    mag_tensor = torch.FloatTensor(mag[:n_seqs * seq_len]).view(n_seqs, seq_len, -1)
    # Shape: (N, Seq, 1)
    f0_tensor = torch.FloatTensor(f0_scaled[:n_seqs * seq_len]).view(n_seqs, seq_len, 1)
    # Shape: (N, Seq, 1)  
    loud_tensor = torch.FloatTensor(loud_scaled[:n_seqs * seq_len]).view(n_seqs, seq_len, 1)
    
    # Create DataLoader
    dataset = TensorDataset(f0_tensor, loud_tensor, mag_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, mag.shape[-1] # returns loader and num_freq_bins

def train(model, dataloader, epochs=50, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Minimal training loop that maps (F0, Loudness) -> Mag_Spectrogram
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For a full DDSP, the model outputs (Harmonics, Noise).
    # We pass those through a synthesizer. 
    # For this simplified prototype (since we don't have custom CUDA synth kernels),
    # we append a linear layer that projects the intermediate DDSP parameters to the full spectrogram.
    # It acts as a neural spectral decoder matching the target shape.
    
    class SpectralDecoder(torch.nn.Module):
        def __init__(self, encoder, freq_bins=1025):
            super().__init__()
            self.encoder = encoder
            # Harmonics (60) + Noise Bands (65) = 125 -> Full FFT Bins (1025)
            self.to_spec = torch.nn.Linear(60 + 65, freq_bins)
            
        def forward(self, f0, loud):
            amp, harmonics, noise = self.encoder(f0, loud)
            x = torch.cat([harmonics, noise], dim=-1)
            raw_spec = torch.nn.functional.relu(self.to_spec(x))
            # Modulate by global amplitude
            return raw_spec * amp
            
    # Assuming freq bins = 1025 (n_fft 2048)
    n_freq_bins = dataloader.dataset.tensors[2].shape[-1]
    
    full_model = SpectralDecoder(model, freq_bins=n_freq_bins).to(device)
    optimizer = optim.Adam(full_model.parameters(), lr=lr)
    criterion = SpectralLoss()

    print(f"\n--- Starting Training on device: {device} ---")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        full_model.train()
        
        for batch_f0, batch_loud, batch_mag in dataloader:
            batch_f0 = batch_f0.to(device)
            batch_loud = batch_loud.to(device)
            batch_mag = batch_mag.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            pred_mag = full_model(batch_f0, batch_loud)
            
            # Loss (MSS uses lin + log)
            loss = criterion(pred_mag, batch_mag)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {avg_loss:.4f}")
            
    return full_model

def main():
    parser = argparse.ArgumentParser(description="☁️ Train a DDSP Voice Model on Ambient Sound")
    parser.add_argument("--base", required=True, help="Path to base audio (e.g. samples/source_clouds.wav)")
    parser.add_argument("--output", default="models/voice.pt", help="Where to save the .pt model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (reduce if out of VRAM)")
    
    args = parser.parse_args()
    
    # 1. Ensure output dir
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 2. Dataset
    loader, n_freq_bins = prepare_dataset(args.base, batch_size=args.batch_size)
    
    # 3. Model
    encoder = MLPDDSPEncoder(n_harmonics=60, n_noise_bands=65)
    
    # 4. Train
    print("\nTraining Neural Voice Cloning Model (DDSP-lite)...")
    print("This learns the 'timbre' and 'texture' of your base audio.")
    
    trained_model = train(encoder, loader, epochs=args.epochs)
    
    # 5. Save
    print(f"\nSaving Voice Model to {args.output}...")
    torch.save(trained_model.state_dict(), args.output)
    print("✅ Training Complete! You can now use this voice in neural inference.")

if __name__ == "__main__":
    main()
