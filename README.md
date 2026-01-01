# 🎵 Two-Input Audio Style Projection

**Transform environmental sounds using vocal characteristics**

*Make the clouds hum Elvis.*

---

## 🌊 What This Project Does

This system takes two audio inputs:
1. **Source Audio** — Environmental sound (wind, rain, clouds, ocean)
2. **Target Audio** — Vocal/musical performance (Elvis singing "Wise Men Say")

And produces an output where the environmental sound *appears to sing* — retaining its natural texture while adopting the melodic and rhythmic qualities of the target.

## 🧠 Core Concepts

### Time-Frequency Representation
Audio is converted to spectrograms using the Short-Time Fourier Transform (STFT), revealing how frequency content evolves over time.

### Content vs. Style
- **Content** = What makes clouds sound like clouds (spectral texture, noise characteristics)
- **Style** = What makes Elvis sound like Elvis singing that song (pitch, rhythm, dynamics)

### The Transformation Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Source      │────▶│ Extract Content  │────▶│             │
│ (Clouds)    │     │ (Texture/Noise)  │     │             │
└─────────────┘     └──────────────────┘     │   BLEND     │────▶ Output
                                             │   (α)       │      (Singing Clouds)
┌─────────────┐     ┌──────────────────┐     │             │
│ Target      │────▶│ Extract Style    │────▶│             │
│ (Elvis)     │     │ (Pitch/Rhythm)   │     └─────────────┘
└─────────────┘     └──────────────────┘
```

## 📁 Project Structure

```
audio-style-projection/
├── notebooks/
│   ├── 01_audio_fundamentals.ipynb    # STFT, spectrograms, basics
│   ├── 02_pitch_extraction.ipynb      # Melody from target audio
│   ├── 03_spectral_modulation.ipynb   # Imprinting melody onto texture
│   ├── 04_full_pipeline.ipynb         # Complete transformation
│   └── 05_experiments.ipynb           # Your playground
├── src/
│   ├── audio_io.py                    # Loading, saving, playback
│   ├── analysis.py                    # STFT, pitch detection
│   ├── transform.py                   # Core transformation algorithms
│   ├── synthesis.py                   # Reconstruction & vocoders
│   └── visualization.py               # Plotting utilities
├── samples/
│   └── (your audio files go here)
├── outputs/
│   └── (generated audio saved here)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Audio Files

Place your audio files in the `samples/` folder:
- `samples/source_clouds.wav` (or any ambient sound)
- `samples/target_elvis.wav` (or any vocal performance)

### 3. Run the Notebooks

```bash
jupyter notebook notebooks/
```

Start with `01_audio_fundamentals.ipynb` and work through sequentially.

## 🎛️ The α Parameter

The blending parameter α controls how much "singing" is applied:

| α Value | Result |
|---------|--------|
| 0.0 | Pure source (just clouds) |
| 0.3 | Subtle rhythmic pulsing |
| 0.5 | Clear melodic movement |
| 0.7 | Strong vocal imprint |
| 1.0 | Maximum transformation |

## 🔬 Technical Deep Dive

### Pitch Contour Extraction
We use `librosa.pyin` to extract the fundamental frequency (F0) over time from the target vocal. This gives us the *melody* as a sequence of frequencies.

### Spectral Modulation
The source audio's spectrum is modulated to emphasize frequencies corresponding to the target's pitch at each moment. This creates the perception of the ambient sound "following" the melody.

### Amplitude Envelope
The dynamics (loud/soft) of the target are transferred to create the breathing, phrasing quality of singing.

### Formant Transfer (Advanced)
Vowel sounds have characteristic frequency peaks (formants). We can add subtle formant-like resonances to give the output a more "vocal" quality.

## 📚 Learning Resources

- [Librosa Documentation](https://librosa.org/doc/)
- [The Science of Sound](https://ccrma.stanford.edu/~jos/)
- [Digital Audio Signal Processing](https://www.dsprelated.com/)

## 🎨 Example Results

*(Add your audio examples here after generating them)*

---

*Built for learning and creative exploration.*

