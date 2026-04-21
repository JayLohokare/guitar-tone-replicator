#!/usr/bin/env python3
"""End-to-end test: generate synthetic guitar, train a tone model, apply it."""
import numpy as np
import soundfile as sf
import requests
import time
import sys

SR = 48000
DURATION = 5  # seconds

# 1. Generate a DI signal (simple clean guitar-like signal: combination of harmonics)
t = np.linspace(0, DURATION, SR * DURATION, dtype=np.float32)
# Fundamental at 330 Hz (E4) with harmonics decaying
di = np.zeros_like(t)
for h in range(1, 8):
    amplitude = 1.0 / h**1.5
    di += amplitude * np.sin(2 * np.pi * 330 * h * t)
# Add some amplitude variation (simulate picking)
envelope = np.exp(-0.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
di = di * envelope / np.max(np.abs(di)) * 0.7

# 2. Create a "target tone" by applying a simple distortion + cab sim effect
target = di.copy()
# Soft clipping (tanh distortion)
target = np.tanh(target * 3.0) * 0.8
# Low-pass filter to simulate cab (simple moving average)
from scipy.signal import lfilter
b = np.ones(20) / 20
target = lfilter(b, 1, target).astype(np.float32)
target = target / np.max(np.abs(target)) * 0.7

# Save files
di_path = "/tmp/test_di.wav"
target_path = "/tmp/test_target_tone.wav"
sf.write(di_path, di, SR)
sf.write(target_path, target, SR)
print(f"✅ Generated DI: {di_path}")
print(f"✅ Generated target tone: {target_path}")

# 3. Start training (skip separation since we have the stem directly)
base_url = "http://localhost:8767"
resp = requests.post(f"{base_url}/train", json={
    "song_path": target_path,
    "model_name": "test_distortion",
    "model_type": "wavenet",
    "model_size": "nano",
    "epochs": 20,
    "skip_separation": True
})
print(f"\n🏋️ Training response: {resp.json()}")
job_id = resp.json().get("job_id")
if not job_id:
    print("❌ Failed to start training")
    sys.exit(1)

# 4. Poll progress
for i in range(60):
    time.sleep(2)
    resp = requests.get(f"{base_url}/progress/{job_id}")
    data = resp.json()
    step = data.get("step", "")
    epoch = data.get("epoch", 0)
    esr = data.get("val_esr", 0)
    if step == "training" and epoch > 0:
        print(f"  Epoch {epoch}: ESR = {esr:.6f}")
    elif step == "complete":
        print(f"\n✅ Training complete! {data.get('message', '')}")
        break
    elif step == "error":
        print(f"\n❌ Training error: {data.get('message', '')}")
        sys.exit(1)

# 5. List models
resp = requests.get(f"{base_url}/models")
models = resp.json()
print(f"\n📋 Trained models: {len(models)}")
for m in models:
    print(f"  - {m['name']}: ESR = {m['best_val_esr']:.6f}, path = {m['model_path']}")

# 6. Apply the model to our DI
if models:
    model = models[0]
    output_path = "/tmp/test_tone_applied.wav"
    resp = requests.post(f"{base_url}/apply", json={
        "model_path": model["model_path"],
        "di_path": di_path,
        "output_path": output_path
    })
    print(f"\n🎸 Apply response: {resp.json()}")
    
    # Verify output exists
    try:
        data, sr = sf.read(output_path)
        print(f"✅ Output file: {output_path}")
        print(f"   Sample rate: {sr}, Duration: {len(data)/sr:.2f}s, Shape: {data.shape}")
    except Exception as e:
        print(f"❌ Failed to read output: {e}")

print("\n🏁 End-to-end test complete!")