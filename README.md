# 🎸 Guitar Tone Replicator

Neural amp modeling — paste a song URL, extract the guitar tone, train a WaveNet model, and play your guitar through that tone in real time.

## How It Works

1. **Paste a URL** — YouTube, SoundCloud, or any audio URL
2. **Extract guitar** — HTDemucs separates the guitar stem from the mix
3. **Train a model** — WaveNet learns the amp/cab tone from the guitar signal
4. **Play live** — Route your guitar through the trained model in real time

## Architecture

### Core Model: WaveNet v2

- Dilated causal convolutions with skip connections
- Zero-initialized output layer for stable training
- Kernel size 5, stacked dilation patterns
- Pre-emphasis weighted ESR + DC loss (MRSTFT removed for MPS performance)

### Model Sizes

| Size    | Parameters | Speed       | Use Case          |
|---------|-----------|-------------|-------------------|
| nano    | 32K       | Fastest     | Quick experiments |
| lite    | 156K      | Fast        | Live play         |
| feather | 228K      | Medium      | Balanced          |
| standard | 433K    | Slower      | Best quality      |

### Best Results (lite, 50 epochs, gain-preserving)

| Metric          | Value  |
|-----------------|--------|
| Raw ESR         | 2.30%  |
| Pre-emph ESR    | 4.97%  |
| Level ratio     | 0.89   |
| Gain ratio      | 3.13x  |
| Correlation     | 0.994  |
| HF ESR (8-16kHz)| 5.5%  |
| LF ESR (0-200Hz)| 33.4% |

## Components

### Python API Server (Port 8767)

FastAPI server handling the full pipeline: download → separate → train → serve models.

```bash
cd ~/dev/guitar-tone-replicator
source venv/bin/activate
python server.py
```

### Guitar Separation API (Port 8766)

HTDemucs-based audio source separation, optimized for Apple Silicon (MLX).

```bash
# v2 (recommended, ~25-30 sec on Apple Silicon)
# Runs on Jay's Mac mini
```

### SwiftUI Mac App

Native macOS app with three tabs:

- **Replicate** — Paste URL, configure training, run the pipeline
- **Live Play** — Real-time guitar processing through trained models
- **Models** — Browse and manage trained tone models

### Real-Time Inference Engine

Python subprocess communicating via stdin/stdout JSON protocol:

- `load` — Load a trained `.pth` model
- `process` — Process audio buffers in real time
- `set_param` — Adjust input gain, output gain, dry/wet mix
- `unload` / `quit` — Clean up

## Setup

### Prerequisites

- Python 3.12+ with venv
- Swift 6.0+ (Xcode) for the Mac app
- Apple Silicon Mac (for MLX-accelerated separation)
- Audio interface (e.g., Scarlett Solo) for live play

### Install

```bash
cd guitar-tone-replicator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Start the API server + Mac app
./start.sh

# Or individually:
source venv/bin/activate
python server.py              # API on :8767
open ToneReplicatorApp/.build/release/ToneReplicatorApp.app  # GUI
```

## Project Structure

```
guitar-tone-replicator/
├── server.py                    # FastAPI pipeline server
├── realtime_inference.py        # Real-time inference subprocess
├── start.sh                     # One-command startup
├── src/
│   └── core/
│       └── model.py             # WaveNet v2 model definition
├── ToneReplicatorApp/           # SwiftUI macOS app
│   ├── Package.swift
│   └── ToneReplicatorApp/
│       └── ToneReplicatorApp.swift  # Full app (UI + audio engine)
├── CoreMLConverter/
│   └── convert_model.py         # PyTorch → CoreML conversion
└── models/                      # Trained model checkpoints
    └── MesaBoogie_v2_lite_gainpres_50ep/
        ├── model.pth
        └── metadata.json
```

## Training Pipeline

1. Download audio from URL (yt-dlp)
2. Separate guitar stem (HTDemucs 6-source model)
3. Align reference (DI) and target (amped) signals
4. Apply gain-preserving normalization
5. Train WaveNet v2 with pre-emphasis weighted ESR loss
6. Save model + metadata for inference

## Live Play

1. Connect an audio interface (Scarlett Solo recommended)
2. Select a trained model from the dropdown
3. Hit **Load** to load the model into the inference engine
4. Hit **Play** to start real-time processing
5. Adjust input gain, output gain, and dry/wet mix in real time
6. Use **Bypass** to compare processed vs dry signal

## Known Issues

- **Output level**: Trained models output ~89% of target level (gain not fully learned)
- **LF ESR**: 0-200Hz range has 33% error (low-frequency response needs work)
- **macOS permission**: App requires microphone access — first launch will prompt, or reset via `tccutil reset Microphone com.tone-replicator.app`

## Tech Stack

- **Model**: PyTorch WaveNet v2
- **Separation**: HTDemucs (PyTorch + MLX)
- **Server**: FastAPI + Uvicorn
- **App**: SwiftUI (Swift 6.0, macOS 14+)
- **Inference**: Real-time subprocess (stdin/stdout protocol)
- **Hardware**: Apple Silicon (M-series) optimized

## License

MIT