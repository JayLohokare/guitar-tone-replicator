#!/usr/bin/env python3
"""
Full training: 100 epochs MesaBoogie Mark V Extreme (paired) + apply to test set.
Produces a high-quality tone model ready for the Mac app.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trainer import ToneTrainer
import soundfile as sf
import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS = os.path.expanduser("~/ToneReplicator/models")

print("🎸 Full Training: MesaBoogie Mark V Extreme")
print("=" * 50)

dry = f"{DATA}/mesaboogie_extreme/DRY/trainval/train.input.wav"
wet = f"{DATA}/mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/trainval/G050/G050.train.target.wav"
test_dry = f"{DATA}/mesaboogie_extreme/DRY/test/test.input.wav"
test_wet = f"{DATA}/mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/test/G050/G050.test.target.wav"

print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
print(f"Epochs: 100, Model: WaveNet lite")
print()

save = f"{MODELS}/MesaBoogie_MarkV_Extreme"
os.makedirs(save, exist_ok=True)

trainer = ToneTrainer(model_type="wavenet", model_size="lite", sample_rate=44100)
start = time.time()
result = trainer.train_paired(di_path=dry, processed_path=wet, epochs=100, save_path=save)
elapsed = time.time() - start

print(f"\n✅ Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"   Best Val ESR: {result['best_val_esr']:.6f}")

# Apply to test set
out_path = f"{save}/test_output.wav"
t2 = ToneTrainer(model_type="wavenet", model_size="lite", sample_rate=44100)
t2.load_model(save)
t2.process_audio(test_dry, out_path)

pred, _ = sf.read(out_path)
truth, _ = sf.read(test_wet)
min_len = min(len(pred), len(truth))
pred, truth = pred[:min_len], truth[:min_len]
test_esr = ((pred - truth) ** 2).sum() / (truth ** 2).sum()
print(f"\n📊 Test ESR: {test_esr:.6f}")
print(f"   Output: {out_path}")
print(f"   Duration: {len(pred)/44100:.2f}s")
print("\n🏁 Model ready for the Mac app!")