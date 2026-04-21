#!/usr/bin/env python3
"""
Real-time inference process for Tone Replicator.
Reads JSON commands from stdin, writes JSON responses + raw audio to stdout.

Protocol:
  1. App sends: {"cmd": "load", "model_path": "/path/to/model_dir"} → {"status": "ok"} or {"status": "error", "message": "..."}
  2. App sends: {"cmd": "process", "sample_rate": 48000, "channels": 1, "frames": N}
     followed by N*4 bytes of float32 PCM audio on stdin
     → {"status": "ok", "frames": N} followed by N*4 bytes of float32 PCM on stdout
  3. App sends: {"cmd": "set_param", "input_gain_db": 0, "output_gain_db": 0, "dry_wet": 100}
  4. App sends: {"cmd": "unload"} → {"status": "ok"}
  5. App sends: {"cmd": "quit"} → exit

All JSON messages are newline-delimited. Audio data follows immediately after the JSON line.
"""

import sys
import json
import struct
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

# Add parent to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent))

from src.core.model import create_model

# Global state
model = None
model_config = {}
device = None
receptive_field = 0

# Parameters
input_gain_db = 0.0
output_gain_db = 0.0
dry_wet = 100.0  # percent

SAMPLE_RATE = 48000


def send_json(obj):
    """Send a JSON response followed by newline."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_json():
    """Read a JSON line from stdin."""
    line = sys.stdin.readline().strip()
    if not line:
        return None
    return json.loads(line)


def read_audio_frames(n_frames):
    """Read n_frames float32 samples from stdin as raw bytes."""
    raw = sys.stdin.buffer.read(n_frames * 4)
    if len(raw) < n_frames * 4:
        return None
    return np.frombuffer(raw, dtype=np.float32)


def load_model(model_path):
    global model, model_config, device, receptive_field
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        return {"status": "error", "message": f"Model directory not found: {model_path}"}
    
    checkpoint_path = model_dir / "model.pth"
    if not checkpoint_path.exists():
        return {"status": "error", "message": f"model.pth not found in {model_path}"}
    
    # Load checkpoint
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load checkpoint: {e}"}
    
    model_type = checkpoint.get("model_type", "wavenet")
    model_size = checkpoint.get("model_size", "standard")
    sr = checkpoint.get("sample_rate", 48000)
    
    # Auto device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create and load model
    model = create_model(model_type, model_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    receptive_field = getattr(model, "receptive_field", 0)
    
    model_config = {
        "model_type": model_type,
        "model_size": model_size,
        "sample_rate": sr,
        "best_val_esr": checkpoint.get("best_val_esr", -1),
        "receptive_field": receptive_field,
        "device": str(device),
    }
    
    return {"status": "ok", "config": model_config}


def process_audio(n_frames, input_audio):
    """Run inference on input audio and return processed audio."""
    global model, device, receptive_field, input_gain_db, output_gain_db, dry_wet
    
    if model is None:
        # No model loaded — passthrough
        return input_audio.copy()
    
    # Apply input gain
    input_gain_linear = 10.0 ** (input_gain_db * 0.05)
    output_gain_linear = 10.0 ** (output_gain_db * 0.05)
    wet_mix = dry_wet / 100.0
    dry_mix = 1.0 - wet_mix
    
    # Convert to tensor: [1, 1, N]
    audio_tensor = torch.tensor(input_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Pad for receptive field
    if receptive_field > 0:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (receptive_field, 0))
    
    with torch.no_grad():
        output = model(audio_tensor)
    
    # Remove receptive field padding
    if receptive_field > 0:
        output = output[:, :, receptive_field:]
    
    # Ensure output length matches input
    output = output.squeeze().cpu().numpy()
    if len(output) > n_frames:
        output = output[:n_frames]
    elif len(output) < n_frames:
        output = np.pad(output, (0, n_frames - len(output)))
    
    # Apply gain and dry/wet mix
    input_scaled = input_audio * input_gain_linear * output_gain_linear
    output_scaled = output * output_gain_linear
    
    mixed = input_scaled * dry_mix + output_scaled * wet_mix
    
    # Soft clip
    mixed = np.tanh(mixed * 2.0) / 2.0
    
    return mixed.astype(np.float32)


def main():
    global input_gain_db, output_gain_db, dry_wet
    
    # Signal ready
    send_json({"status": "ready", "pid": __import__("os").getpid()})
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            cmd = json.loads(line)
            action = cmd.get("cmd", "")
            
            if action == "load":
                result = load_model(cmd["model_path"])
                send_json(result)
            
            elif action == "process":
                n_frames = cmd.get("frames", 0)
                sample_rate = cmd.get("sample_rate", SAMPLE_RATE)
                
                if n_frames <= 0:
                    send_json({"status": "error", "message": "Invalid frame count"})
                    continue
                
                input_audio = read_audio_frames(n_frames)
                if input_audio is None:
                    send_json({"status": "error", "message": "Failed to read audio data"})
                    continue
                
                output_audio = process_audio(n_frames, input_audio)
                
                send_json({"status": "ok", "frames": n_frames})
                sys.stdout.buffer.write(output_audio.tobytes())
                sys.stdout.buffer.flush()
            
            elif action == "set_param":
                if "input_gain_db" in cmd:
                    input_gain_db = cmd["input_gain_db"]
                if "output_gain_db" in cmd:
                    output_gain_db = cmd["output_gain_db"]
                if "dry_wet" in cmd:
                    dry_wet = cmd["dry_wet"]
                send_json({"status": "ok"})
            
            elif action == "unload":
                global model
                model = None
                model_config = {}
                send_json({"status": "ok"})
            
            elif action == "quit":
                send_json({"status": "ok"})
                break
            
            else:
                send_json({"status": "error", "message": f"Unknown command: {action}"})
        
        except json.JSONDecodeError as e:
            send_json({"status": "error", "message": f"Invalid JSON: {e}"})
        except Exception as e:
            send_json({"status": "error", "message": str(e)})
    
    # Cleanup
    if model is not None:
        global device
        del model
        if device is not None and device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()