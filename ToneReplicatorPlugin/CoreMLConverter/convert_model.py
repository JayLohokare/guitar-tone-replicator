#!/usr/bin/env python3
"""
Tone Replicator - PyTorch to CoreML Model Converter
Converts trained .pth model files to .mlpackage format for AUv3 plugin inference.

Requirements: Python 3.12+ (3.14 doesn't support coremltools native libs)
Install: pip install torch coremltools numpy

Usage:
    python convert_model.py --input model.pth --output model.mlpackage --name MyAmp
    python convert_model.py --input model.pth --output_dir ~/ToneReplicator/models/MyAmp/
    python convert_model.py --dir ~/ToneReplicator/models/  # Convert all models in directory

MIT License
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Inline model definition (avoids dependency on ToneReplicator project)

class ConvBlock(nn.Module):
    """Dilated causal convolution block with gated activation."""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels * 2, kernel_size, dilation=dilation,
                      padding=(kernel_size - 1) * dilation)
        )
        self.residual = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h[:, :, :x.shape[-1]]  # Causal: trim future
        gate, filter_ = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filter_)
        return self.residual(h) + x


class ToneNet(nn.Module):
    """WaveNet-style tone model."""
    
    def __init__(self, channels=24, num_blocks=6, kernel_size=3, dilations=None, sample_rate=44100.0):
        super().__init__()
        self.sample_rate = sample_rate
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32][:num_blocks]
        self.input_conv = nn.utils.weight_norm(nn.Conv1d(1, channels, 1))
        self.blocks = nn.ModuleList([ConvBlock(channels, kernel_size, d) for d in dilations])
        self.output_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels, channels, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(channels, 1, 1)),
        )
        self._receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)

    @property
    def receptive_field(self):
        return self._receptive_field

    def forward(self, x):
        pad = self._receptive_field - 1
        x = F.pad(x, (pad, 0))
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h)
        h = h[:, :, pad:]
        return self.output_conv(h)


# Model configurations matching ToneReplicator's src/core/model.py
CONFIGS = {
    "wavenet": {
        "nano": {"channels": 16, "num_blocks": 4, "kernel_size": 3, "dilations": [1, 2, 4, 8]},
        "lite": {"channels": 24, "num_blocks": 6, "kernel_size": 3, "dilations": [1, 2, 4, 8, 16, 32]},
        "standard": {"channels": 32, "num_blocks": 10, "kernel_size": 3, "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
        "feather": {"channels": 24, "num_blocks": 8, "kernel_size": 3, "dilations": [1, 2, 4, 8, 16, 32, 64, 128]},
    }
}


def load_model(model_path):
    """Load a ToneNet model from a .pth checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "wavenet")
    model_size = checkpoint.get("model_size", "lite")
    sample_rate = checkpoint.get("sample_rate", 44100.0)
    
    config = CONFIGS.get(model_type, CONFIGS["wavenet"]).get(model_size, CONFIGS["wavenet"]["lite"])
    config["sample_rate"] = sample_rate
    
    model = ToneNet(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, model_type, model_size, sample_rate


def remove_weight_norm(model):
    """Remove weight normalization for clean JIT tracing."""
    for module in model.modules():
        if hasattr(module, 'weight_g') or hasattr(module, 'weight_v'):
            try:
                torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
            except Exception:
                pass


def convert_to_coreml(model, output_path, sample_rate=44100.0, chunk_size=8192):
    """Convert a PyTorch model to CoreML format."""
    import coremltools as ct
    
    # Remove weight normalization for clean tracing
    remove_weight_norm(model)
    
    # Trace the model
    sample_input = torch.randn(1, 1, chunk_size)
    print(f"Tracing model with input shape {sample_input.shape}...")
    traced = torch.jit.trace(model, sample_input)
    
    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=sample_input.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    
    # Add metadata
    mlmodel.short_description = "Tone Replicator - Neural Amp Model"
    mlmodel.input_description["input"] = "Mono audio input (1, 1, samples)"
    mlmodel.output_description["output"] = "Processed audio output (1, 1, samples)"
    
    # Save
    print(f"Saving CoreML model to {output_path}...")
    mlmodel.save(output_path)
    print("Conversion complete!")
    return mlmodel


def convert_single_model(input_path, output_path=None, name=None):
    """Convert a single .pth model to CoreML."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path.parent / "model.mlpackage"
    
    model, model_type, model_size, sample_rate = load_model(input_path)
    print(f"Model: {model_type}/{model_size}, sample_rate={sample_rate}")
    print(f"Receptive field: {model.receptive_field}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    convert_to_coreml(model, str(output_path), sample_rate=sample_rate)
    return output_path


def convert_all_in_directory(directory):
    """Convert all .pth models that don't already have .mlpackage."""
    directory = Path(directory)
    converted = []
    
    for model_dir in sorted(directory.iterdir()):
        if not model_dir.is_dir():
            continue
        
        pth_path = model_dir / "model.pth"
        mlpackage_path = model_dir / "model.mlpackage"
        
        if not pth_path.exists():
            continue
        
        if mlpackage_path.exists():
            print(f"Skipping {model_dir.name} (already converted)")
            continue
        
        print(f"\nConverting {model_dir.name}...")
        try:
            result = convert_single_model(pth_path)
            converted.append((model_dir.name, result))
        except Exception as e:
            print(f"Failed to convert {model_dir.name}: {e}")
    
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert Tone Replicator models to CoreML")
    parser.add_argument("--input", "-i", help="Path to .pth model file")
    parser.add_argument("--output", "-o", help="Output path for .mlpackage")
    parser.add_argument("--name", "-n", help="Model name")
    parser.add_argument("--dir", "-d", help="Convert all models in directory")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size for tracing")
    args = parser.parse_args()
    
    if args.dir:
        convert_all_in_directory(args.dir)
    elif args.input:
        convert_single_model(args.input, args.output, args.name)
    else:
        default_dir = Path.home() / "ToneReplicator" / "models"
        if default_dir.exists():
            convert_all_in_directory(default_dir)
        else:
            parser.print_help()
            print("\nError: No input specified and default directory not found.")


if __name__ == "__main__":
    main()