"""
Neural Inference Pipeline (AI Voice Cloning)

Reads a trained "voice" model (e.g. clouds.pt) and a script (e.g. target_elvis.wav).
Extracts F0/Loudness from Elvis, feeds it to the Cloud.pt model, and outputs Singing Clouds.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import librosa

# Support running from root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.audio_io import load_audio, save_audio
from src.analysis import extract_pitch, extract_envelope
from src.neural_core import MLPDDSPEncoder, scale_features

# We need the SpectralDecoder class we built in training to load the weights
class SpectralDecoder(torch.nn.Module):
    def __init__(self, encoder, freq_bins=1025):
        super().__init__()
        self.encoder = encoder
        self.to_spec = torch.nn.Linear(60 + 65, freq_bins)
        
    def forward(self, f0, loud):
        amp, harmonics, noise = self.encoder(f0, loud)
        x = torch.cat([harmonics, noise], dim=-1)
        raw_spec = torch.nn.functional.relu(self.to_spec(x))
        return raw_spec * amp

def neural_transform(
    model_path: str,
    projected_path: str,
    output_path: str = "outputs/neural_output.wav",
    sr: int = 22050,
    hop_length: int = 256,
    n_fft: int = 2048,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("============================================================")
    print("  🧠 Neural Voice Style Projection (DDSP Inference)")
    print("============================================================")
    print(f"  Voice Model : {model_path}")
    print(f"  Singer      : {projected_path}")
    print(f"  Output      : {output_path}")
    print("============================================================")
    
    # 1. Load the Voice Model
    print(f"[NEURAL] Loading AI Voice Model to {device}...")
    encoder = MLPDDSPEncoder(n_harmonics=60, n_noise_bands=65)
    # n_fft=2048 -> freq_bins=1025
    model = SpectralDecoder(encoder, freq_bins=(n_fft // 2 + 1))
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    # 2. Read the "Script" (Elvis)
    print(f"[NEURAL] Reading Singer (the 'Script')...")
    y_proj, _ = load_audio(projected_path, sr=sr)
    
    # 3. Extract F0 and Loudness
    print(f"[NEURAL] Extracting Melody (F0) & Rhythm (Loudness)...")
    f0, voiced, _ = extract_pitch(y_proj, sr, hop_length=hop_length)
    loudness, _ = extract_envelope(y_proj, sr, hop_length=hop_length)
    
    # Ensure matching lengths
    n_frames = min(len(f0), len(loudness))
    f0 = f0[:n_frames]
    loudness = loudness[:n_frames]
    
    # 4. Scale inputs for Neural Net
    f0_scaled, loud_scaled = scale_features(f0, loudness)
    
    # 5. Run Inference!
    print(f"[NEURAL] Running AI Inference (Synthesizing Spectrogram)...")
    with torch.no_grad():
        f0_tensor = torch.FloatTensor(f0_scaled).view(1, -1, 1).to(device)
        loud_tensor = torch.FloatTensor(loud_scaled).view(1, -1, 1).to(device)
        
        pred_mag_tensor = model(f0_tensor, loud_tensor)
        # shape: (1, time, freq_bins) -> (freq_bins, time)
        pred_mag = pred_mag_tensor.squeeze(0).cpu().numpy().T
    
    # 6. Vocoder Reconstruction (Griffin-Lim)
    # We predicted the magnitude spectrogram. Now we need Phase to convert to audio.
    print(f"[NEURAL] Reconstructing Audio (Griffin-Lim Phase Estimation)...")
    
    # Give the audio back its correct shape, scaling
    pred_mag = pred_mag * 1.5 # slight boost
    
    # Griffin-lim
    y_output = librosa.griffinlim(pred_mag, n_iter=64, hop_length=hop_length, win_length=n_fft)
    
    # Optionally: Apply the original Singer Phase if it helps coherence (Phase-Vocoder trick)
    # S_proj = librosa.stft(y_proj, n_fft=n_fft, hop_length=hop_length)
    # phase_proj = np.angle(S_proj[:, :pred_mag.shape[1]])
    # S_output = pred_mag * np.exp(1j * phase_proj)
    # y_output = librosa.istft(S_output, hop_length=hop_length)
    
    # Normalize
    peak = np.max(np.abs(y_output))
    if peak > 0:
        y_output = (y_output / peak) * 0.95
        
    # Save
    save_audio(output_path, y_output, sr)
    print(f"\n✅ Neural Generation Complete! Listen to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🧠 Neural Audio Inference (DDSP)")
    parser.add_argument("--model", required=True, help="Path to trained .pt voice model")
    parser.add_argument("--projected", required=True, help="Path to singer (script) audio")
    parser.add_argument("--output", default="outputs/neural_output.wav", help="Output path")
    
    args = parser.parse_args()
    
    neural_transform(args.model, args.projected, args.output)
