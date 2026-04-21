"""
Quick test to verify the Tone Replicator core model trains and infers.
"""

import sys
import os
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model import create_model
from src.core.dataset import ReferenceDIProvider
from src.core.trainer import ToneTrainer


def test_model_creation():
    """Test that models can be created."""
    print("Testing model creation...")
    
    for model_type in ["wavenet", "lstm"]:
        for size in ["nano", "lite"]:
            model = create_model(model_type, size)
            print(f"  {model_type}/{size}: {sum(p.numel() for p in model.parameters())} params")
            assert model is not None
    
    print("✓ Model creation passed")


def test_forward_pass():
    """Test forward pass through the model."""
    print("\nTesting forward pass...")
    
    import torch
    
    model = create_model("wavenet", "nano")
    model.eval()
    
    # Create dummy input: batch=1, channels=1, time=8192
    x = torch.randn(1, 1, 8192)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    print("✓ Forward pass passed")


def test_reference_di_generation():
    """Test reference DI signal generation."""
    print("\nTesting reference DI generation...")
    
    provider = ReferenceDIProvider(sample_rate=48000, duration_seconds=10.0)
    signal = provider.generate()
    
    print(f"  Signal length: {len(signal)} samples ({len(signal)/48000:.1f}s)")
    print(f"  Max amplitude: {np.max(np.abs(signal)):.4f}")
    print(f"  RMS: {np.sqrt(np.mean(signal**2)):.4f}")
    
    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "test_ref_di.wav")
    provider.save(temp_path)
    print(f"  Saved to: {temp_path}")
    
    # Verify we can read it back
    loaded, sr = sf.read(temp_path)
    print(f"  Reloaded: {len(loaded)} samples at {sr}Hz")
    
    print("✓ Reference DI generation passed")


def test_quick_training():
    """Test a quick training cycle with synthetic data."""
    print("\nTesting quick training (2 epochs)...")
    
    import torch
    
    # Create synthetic DI + processed pair
    sr = 48000
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # DI: clean guitar-like signal
    di = np.sin(2 * np.pi * 220 * t) * 0.5 + np.sin(2 * np.pi * 440 * t) * 0.3
    di = di * np.exp(-2.0 * t)  # Decay
    di = di / np.max(np.abs(di)) * 0.8
    
    # "Processed": add some distortion and filtering (simulating amp)
    processed = np.tanh(di * 3.0) * 0.7  # Soft clipping
    processed = processed + np.sin(2 * np.pi * 50 * t) * 0.02  # Add some low-end
    processed = processed / np.max(np.abs(processed)) * 0.8
    
    # Save to temp files
    temp_dir = tempfile.mkdtemp()
    di_path = os.path.join(temp_dir, "di.wav")
    proc_path = os.path.join(temp_dir, "processed.wav")
    sf.write(di_path, di, sr)
    sf.write(proc_path, processed, sr)
    
    # Train
    trainer = ToneTrainer(
        model_type="wavenet",
        model_size="nano",
        learning_rate=0.01,
    )
    
    result = trainer.train_paired(
        di_path=di_path,
        processed_path=proc_path,
        epochs=2,
        save_path=os.path.join(temp_dir, "test_model"),
    )
    
    print(f"  Training result: best_val_esr = {result['best_val_esr']:.4f}")
    print(f"  Epochs trained: {len(result['history'])}")
    
    # Test inference
    trainer.load_model(os.path.join(temp_dir, "test_model"))
    output_path = os.path.join(temp_dir, "test_output.wav")
    trainer.process_audio(di_path, output_path)
    
    # Verify output
    output, out_sr = sf.read(output_path)
    print(f"  Output: {len(output)} samples at {out_sr}Hz")
    
    print("✓ Quick training passed")


if __name__ == "__main__":
    print("=" * 60)
    print("🎸 Tone Replicator - Core Tests")
    print("=" * 60)
    
    test_model_creation()
    test_forward_pass()
    test_reference_di_generation()
    test_quick_training()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)