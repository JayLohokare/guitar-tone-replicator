#!/usr/bin/env python3
"""
Convert a trained PyTorch tone model to CoreML format.
Requires Python 3.12 (coremltools binary compatibility).
Usage: python3.12 convert_model.py --model-dir ~/ToneReplicator/models/MesaBoogie_MarkV_Extreme
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model import create_model


def convert(model_dir: str, sample_duration: float = 2.0):
    """Convert PyTorch model to CoreML .mlpackage."""
    model_dir = Path(model_dir)
    
    # Load checkpoint
    checkpoint = torch.load(
        model_dir / "model.pth",
        map_location="cpu",
        weights_only=True,
    )
    
    # Create and load model
    model = create_model(checkpoint["model_type"], checkpoint["model_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    sample_rate = checkpoint.get("sample_rate", 44100)
    num_samples = int(sample_rate * sample_duration)
    
    print(f"Model: {checkpoint['model_type']}/{checkpoint['model_size']}")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Input shape: (1, 1, {num_samples})")
    print(f"Best Val ESR: {checkpoint.get('best_val_esr', 'N/A')}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trace model
    print("\nTracing model...")
    sample_input = torch.randn(1, 1, num_samples)
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
    
    # Save
    coreml_path = model_dir / "model.mlpackage"
    mlmodel.save(str(coreml_path))
    print(f"\n✅ CoreML model saved to: {coreml_path}")
    
    # Update metadata
    metadata = {}
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata["coreml_path"] = str(coreml_path)
    metadata["coreml_input_samples"] = num_samples
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Verify
    print("\nVerifying CoreML model...")
    result = mlmodel.predict({"input": np.random.randn(1, 1, num_samples).astype(np.float32)})
    output_shape = result["output"].shape
    print(f"Output shape: {output_shape}")
    print(f"✅ Model verified and ready for AUv3 plugin!")
    
    return str(coreml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch tone model to CoreML")
    parser.add_argument("--model-dir", required=True, help="Path to model directory with model.pth")
    parser.add_argument("--duration", type=float, default=2.0, help="Sample duration in seconds for model input")
    args = parser.parse_args()
    
    convert(args.model_dir, args.duration)