#!/usr/bin/env python3
"""Quick real-data training test — 30 epochs, MesaBoogie Mark V Extreme."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trainer import ToneTrainer
import soundfile as sf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS = os.path.expanduser("~/ToneReplicator/models")

print("🎸 Quick Real Data Training Test")
print("=" * 50)

# MesaBoogie Mark V Extreme — paired mode
dry = f"{DATA}/mesaboogie_extreme/DRY/trainval/train.input.wav"
wet = f"{DATA}/mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/trainval/G050/G050.train.target.wav"

# Verify files
d, sr = sf.read(dry); print(f"Dry: {len(d)/sr:.1f}s @ {sr}Hz")
d, sr = sf.read(wet); print(f"Wet: {len(d)/sr:.1f}s @ {sr}Hz")

import torch
print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
print()

# Train — 30 epochs, lite model
save = f"{MODELS}/MesaBoogie_MarkV_Extreme_paired"
os.makedirs(save, exist_ok=True)

trainer = ToneTrainer(model_type="wavenet", model_size="lite", sample_rate=44100)
start = time.time()
result = trainer.train_paired(di_path=dry, processed_path=wet, epochs=30, save_path=save)
elapsed = time.time() - start

print(f"\n✅ Training complete in {elapsed:.1f}s")
print(f"   Best Val ESR: {result['best_val_esr']:.6f}")

# Apply to test set
test_dry = f"{DATA}/mesaboogie_extreme/DRY/test/test.input.wav"
test_wet = f"{DATA}/mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/test/G050/G050.test.target.wav"
out_path = f"{MODELS}/MesaBoogie_MarkV_Extreme_paired/test_output.wav"

if os.path.exists(test_dry):
    trainer2 = ToneTrainer(model_type="wavenet", model_size="lite", sample_rate=44100)
    trainer2.load_model(save)
    trainer2.process_audio(test_dry, out_path)
    
    # Compare output with ground truth
    pred, _ = sf.read(out_path)
    truth, _ = sf.read(test_wet)
    min_len = min(len(pred), len(truth))
    pred = pred[:min_len]
    truth = truth[:min_len]
    esr = ((pred - truth) ** 2).sum() / (truth ** 2).sum()
    print(f"\n📊 Test set ESR: {esr:.6f}")
    print(f"   Output: {out_path}")
    print(f"   Duration: {len(pred)/44100:.2f}s")

print("\n🏁 Done!")