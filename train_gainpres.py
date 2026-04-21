#!/usr/bin/env python3
"""Train with gain-preserving normalization."""
from src.core.trainer import ToneTrainer
import time

trainer = ToneTrainer(
    model_type='wavenet',
    model_size='lite',
    sample_rate=44100,
    segment_length=8192,
    device='mps',
    batch_size=16,
    learning_rate=0.003,
)

di_path = 'data/mesaboogie_extreme/DRY/trainval/train.input.wav'
target_path = 'data/mesaboogie_extreme/MesaBoogie-MarkV-ChExtreme/trainval/G050/G050.train.target.wav'

print('Training LITE v2 (50 epochs, gain-preserving)...', flush=True)
start = time.time()
result = trainer.train_paired(
    di_path=di_path,
    processed_path=target_path,
    epochs=50,
    save_path='/Users/jaylohokare/ToneReplicator/models/MesaBoogie_v2_lite_gainpres_50ep',
)
elapsed = time.time() - start
print(f'Done in {elapsed:.0f}s! Best val ESR: {result["best_val_esr"]:.6f}', flush=True)
for h in result['history']:
    print(f'  Epoch {h["epoch"]}: train_esr={h["train_esr"]:.6f}, val_esr={h["val_esr"]:.6f}', flush=True)