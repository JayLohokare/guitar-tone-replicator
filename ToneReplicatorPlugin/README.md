# Tone Replicator AUv3 Plugin

Neural amp modeler Audio Unit v3 plugin for macOS. Runs trained WaveNet tone models in real-time using CoreML for Apple Silicon optimization.

## Architecture

- **Inference Engine**: CoreML (converted from PyTorch `.pth` → `.mlpackage`)
- **Plugin Framework**: Apple AUv3 APIs (no iPlug2 or GPL dependencies)
- **DSP Kernel**: C++ with Swift bridge (real-time safe, no allocations in render loop)
- **Standalone App**: SwiftUI with AVAudioEngine for standalone usage
- **Model**: WaveNet "lite" (24 channels, 6 dilated blocks, 127-sample receptive field, ~25K params)

## Project Structure

```
ToneReplicatorPlugin/
├── Package.swift                           # Swift Package Manager build
├── project.yml                             # XcodeGen config (for AUv3 extension target)
├── CoreMLConverter/
│   └── convert_model.py                   # PyTorch → CoreML conversion script
├── Sources/
│   ├── ToneReplicatorApp/                 # Standalone macOS app (SwiftUI)
│   │   ├── ToneReplicatorPluginApp.swift   # App entry point
│   │   ├── ContentView.swift              # Main UI (model browser, controls, meters)
│   │   ├── ModelStore.swift              # Model discovery and management
│   │   └── AudioEngineManager.swift       # AVAudioEngine + CoreML inference
│   ├── ToneReplicatorAU/                  # AUv3 extension (for DAW hosting)
│   │   ├── ToneReplicatorAU.swift         # AUAudioUnit subclass
│   │   └── ToneReplicatorBridge.swift      # Swift-C++ bridge declarations
│   └── ToneReplicatorDSP/                 # C++ real-time DSP kernel
│       ├── ToneReplicatorKernel.hpp        # Kernel header
│       ├── ToneReplicatorKernel.cpp        # Kernel implementation
│       └── ToneReplicatorKernelBridge.cpp  # C bridge for Swift interop
├── ToneReplicatorPlugin/                   # App extension host (Xcode project files)
│   ├── Info.plist
│   ├── ToneReplicatorPlugin.entitlements
│   └── *.swift                             # (duplicate of Sources/ for xcodeproj)
└── ToneReplicatorAU/                       # AU extension (Xcode project files)
    ├── Info.plist
    ├── ToneReplicatorAU.entitlements
    └── *.swift, *.cpp, *.hpp              # (duplicate of Sources/ for xcodeproj)
```

## Building

### Swift Package Manager (Standalone App)

```bash
cd ToneReplicatorPlugin

# Debug build
swift build

# Release build
swift build -c release

# Binary locations
# Debug:   .build/debug/ToneReplicatorApp
# Release: .build/release/ToneReplicatorApp
```

### Xcode (Full AUv3 Plugin)

Requires Xcode (not just Command Line Tools).

```bash
# Generate Xcode project from project.yml
xcodegen generate

# Open in Xcode
open ToneReplicatorPlugin.xcodeproj

# Build and run from Xcode
```

The Xcode project builds both:
1. **ToneReplicatorPlugin** - Standalone app host
2. **ToneReplicatorAU** - AUv3 extension (appears in GarageBand, Logic Pro, etc.)

## Model Conversion

Convert trained `.pth` models to CoreML `.mlpackage` format:

```bash
# Requires Python 3.12+ (coremltools doesn't fully support 3.14 yet)
# Using the project's venv:
cd ~/dev/guitar-tone-replicator
source venv/bin/activate

# Convert a specific model
python ToneReplicatorPlugin/CoreMLConverter/convert_model.py \
    --input ~/ToneReplicator/models/MesaBoogie_MarkV_Extreme/model.pth \
    --output ~/ToneReplicator/models/MesaBoogie_MarkV_Extreme/model.mlpackage

# Convert all models in directory
python ToneReplicatorPlugin/CoreMLConverter/convert_model.py \
    --dir ~/ToneReplicator/models/
```

### Known Issue: coremltools on Python 3.14

The `BlobWriter` native library in coremltools 9.0 doesn't work on Python 3.14. If you encounter `RuntimeError: BlobWriter not loaded`, use Python 3.12:

```bash
# Create a Python 3.12 venv for conversion
python3.12 -m venv /tmp/coreml-venv
/tmp/coreml-venv/bin/pip install torch coremltools numpy

# Run conversion with Python 3.12
/tmp/coreml-venv/bin/python ToneReplicatorPlugin/CoreMLConverter/convert_model.py --dir ~/ToneReplicator/models/
```

## Model Format

- **Input**: `.pth` (PyTorch checkpoint with `model_state_dict`, `model_type`, `model_size`, `sample_rate`)
- **Output**: `.mlpackage` (CoreML ML Program, macOS 14+, uses ANE on Apple Silicon)
- **Model Config (lite)**: 24 channels, 6 blocks, dilations [1,2,4,8,16,32], receptive field = 127 samples (~2.9ms at 44.1kHz)

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Input Gain | -40 to +40 dB | 0 dB | Pre-model gain |
| Output Gain | -40 to +40 dB | 0 dB | Post-model gain |
| Dry/Wet Mix | 0-100% | 100% | Blend between dry input and wet processed signal |
| Bypass | On/Off | Off | Zero-latency passthrough |

## License

MIT License - All code is original. No GPL dependencies.

- Apple frameworks (AVFoundation, CoreAudio, CoreML) - commercial-friendly
- coremltools - MIT license ✓
- Our own model code (MIT) ✓

## Status

- [x] C++ DSP kernel (real-time safe)
- [x] Swift-C++ bridge
- [x] AUv3 Audio Unit subclass
- [x] Standalone SwiftUI app
- [x] Model discovery and loading
- [x] CoreML model conversion (PyTorch → CoreML)
- [x] SPM build (standalone app compiles for arm64)
- [x] Verified CoreML model (MesaBoogie Mark V Extreme converted successfully)
- [ ] Full AUv3 extension build (requires Xcode)
- [ ] Real-time CoreML inference in audio render callback
- [ ] DAW testing (GarageBand, Logic Pro)
- [ ] App Store distribution setup