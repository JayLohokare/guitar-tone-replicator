#!/usr/bin/env python3
"""
Real dataset training test using ToneTwisT-AFx paired dry/wet audio.

Tests two models:
1. MesaBoogie Mark V (Extreme channel) - high-gain amp
2. Electro Harmonix Big Muff - classic fuzz pedal

With both paired mode (has DI) and blind mode (only wet).
"""
import sys
import os
import time
import json
import numpy as np
import soundfile as sf

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trainer import ToneTrainer
from src.core.model import create_model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS_DIR = os.path.expanduser("~/ToneReplicator/models")

def train_paired(name, di_path, wet_path, epochs=100, model_size="standard"):
    """Train with paired DI + wet audio (gold standard)."""
    print(f"\n{'='*60}")
    print(f"🎸 PAIRED TRAINING: {name}")
    print(f"   DI:   {di_path}")
    print(f"   Wet:  {wet_path}")
    print(f"   Epochs: {epochs}, Size: {model_size}")
    print(f"{'='*60}")
    
    trainer = ToneTrainer(
        model_type="wavenet",
        model_size=model_size,
        sample_rate=44100,  # Dataset is 44.1kHz
        device="auto",
    )
    
    save_path = os.path.join(MODELS_DIR, name)
    os.makedirs(save_path, exist_ok=True)
    
    start = time.time()
    result = trainer.train_paired(
        di_path=di_path,
        processed_path=wet_path,
        epochs=epochs,
        save_path=save_path,
    )
    elapsed = time.time() - start
    
    print(f"\n✅ {name} trained in {elapsed:.1f}s")
    print(f"   Best Val ESR: {result['best_val_esr']:.6f}")
    print(f"   Model type: {result['model_type']}/{result['model_size']}")
    print(f"   Sample rate: {result['sample_rate']}Hz")
    
    # Apply to test input
    test_di = di_path.replace("train.input", "test.input") if "train.input" in di_path else di_path
    output_path = os.path.join(MODELS_DIR, name, "test_output.wav")
    
    if os.path.exists(test_di):
        trainer.load_model(save_path)
        trainer.process_audio(test_di, output_path)
        d, sr = sf.read(output_path)
        print(f"   Test output: {output_path} ({len(d)/sr:.2f}s)")
    
    return result

def train_blind(name, wet_path, epochs=100, model_size="standard"):
    """Train with only wet audio (blind capture mode)."""
    print(f"\n{'='*60}")
    print(f"🎸 BLIND TRAINING: {name}")
    print(f"   Wet:  {wet_path}")
    print(f"   Epochs: {epochs}, Size: {model_size}")
    print(f"{'='*60}")
    
    trainer = ToneTrainer(
        model_type="wavenet",
        model_size=model_size,
        sample_rate=44100,
        device="auto",
    )
    
    save_path = os.path.join(MODELS_DIR, name)
    os.makedirs(save_path, exist_ok=True)
    
    start = time.time()
    result = trainer.train_blind(
        target_tone_path=wet_path,
        epochs=epochs,
        save_path=save_path,
    )
    elapsed = time.time() - start
    
    print(f"\n✅ {name} trained in {elapsed:.1f}s")
    print(f"   Best Val ESR: {result['best_val_esr']:.6f}")
    
    return result

def main():
    print("🎸 Tone Replicator - Real Dataset Training Test")
    print("=" * 60)
    
    # Check MPS
    import torch
    device = "MPS" if torch.backends.mps.is_available() else "CPU"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # ===== Test 1: MesaBoogie Mark V Extreme - PAIRED =====
    mesa_dry = os.path.join(DATA_DIR, "mesaboogie_extreme/DRY/trainval/train.input.wav")
    mesa_wet = os.path.join(DATA_DIR, "mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/trainval/G050/G050.train.target.wav")
    
    if os.path.exists(mesa_dry) and os.path.exists(mesa_wet):
        print("\n📊 MesaBoogie Mark V Extreme (paired mode)")
        result1 = train_paired(
            "MesaBoogie_MarkV_Extreme_paired",
            mesa_dry, mesa_wet,
            epochs=50,
            model_size="lite",
        )
    else:
        print(f"⚠️ MesaBoogie files not found, skipping paired test")
    
    # ===== Test 2: Big Muff - PAIRED =====
    bm_dry = os.path.join(DATA_DIR, "bigmuff/DRY/input.wav")
    bm_wet = os.path.join(DATA_DIR, "bigmuff/DIY-ElectroHarmonix-BigMuff/Vol=6_Tone=2_Sustain=5/target.wav")
    
    if os.path.exists(bm_dry) and os.path.exists(bm_wet):
        print("\n📊 Electro Harmonix Big Muff (paired mode)")
        result2 = train_paired(
            "BigMuff_paired",
            bm_dry, bm_wet,
            epochs=50,
            model_size="lite",
        )
    else:
        print(f"⚠️ Big Muff files not found, skipping paired test")
    
    # ===== Test 3: MesaBoogie - BLIND =====
    if os.path.exists(mesa_wet):
        print("\n📊 MesaBoogie Mark V Extreme (blind mode - no DI)")
        result3 = train_blind(
            "MesaBoogie_MarkV_Extreme_blind",
            mesa_wet,
            epochs=50,
            model_size="lite",
        )
    
    # ===== Summary =====
    print(f"\n\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    import glob
    models = glob.glob(os.path.join(MODELS_DIR, "*", "metadata.json"))
    for m in models:
        with open(m) as f:
            meta = json.load(f)
        name = os.path.dirname(m).split("/")[-1]
        print(f"  {name}: ESR={meta.get('best_val_esr', 'N/A'):.6f}, "
              f"type={meta.get('model_type')}/{meta.get('model_size')}, "
              f"sr={meta.get('sample_rate')}Hz")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    print("🏁 All tests complete!")

if __name__ == "__main__":
    main()