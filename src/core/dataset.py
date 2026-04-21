"""
Tone Replicator - Dataset v2
Handles paired DI/processed audio for training.

Key improvements over v1:
- Longer default segment length (16384 = ~340ms @ 48kHz, ~372ms @ 44.1kHz)
- Better silence trimming
- Input normalization (peak normalize both channels)
- Option for overlapping segments
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class ToneDataset(Dataset):
    """
    Dataset of (DI, processed) audio pairs for tone modeling.
    
    Splits audio into fixed-length windows for training.
    Uses longer segments than v1 for better temporal modeling.
    """
    
    def __init__(
        self,
        di_path: str,
        processed_path: str,
        sample_rate: int = 48000,
        segment_length: int = 16384,
        delay: int = 0,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ):
        self.segment_length = segment_length
        self.delay = delay
        
        # Load audio
        di_audio, sr1 = librosa.load(di_path, sr=sample_rate, mono=True)
        proc_audio, sr2 = librosa.load(processed_path, sr=sample_rate, mono=True)
        
        # Apply segment bounds
        if start_sample is not None:
            di_audio = di_audio[start_sample:]
            proc_audio = proc_audio[start_sample:]
        if stop_sample is not None:
            di_audio = di_audio[:stop_sample]
            proc_audio = proc_audio[:stop_sample]
        
        # Match lengths
        min_len = min(len(di_audio), len(proc_audio))
        di_audio = di_audio[:min_len]
        proc_audio = proc_audio[:min_len]
        
        # Apply delay compensation
        if delay > 0:
            # Output is delayed relative to input
            proc_audio = proc_audio[delay:]
            di_audio = di_audio[:-delay] if delay < len(di_audio) else di_audio
        elif delay < 0:
            # Input is delayed relative to output
            di_audio = di_audio[-delay:]
            proc_audio = proc_audio[:delay] if -delay < len(proc_audio) else proc_audio
        
        # Match again after delay
        min_len = min(len(di_audio), len(proc_audio))
        di_audio = di_audio[:min_len]
        proc_audio = proc_audio[:min_len]
        
        # Remove initial silence more aggressively
        # Find first significant sample in the processed signal
        threshold = 0.005 * np.max(np.abs(proc_audio))
        start_idx = 0
        significant = np.where(np.abs(proc_audio) > threshold)[0]
        if len(significant) > 0:
            start_idx = max(0, significant[0] - sample_rate // 10)  # 100ms before first note
            start_idx = min(start_idx, sample_rate)  # Don't skip more than 1s
        
        di_audio = di_audio[start_idx:]
        proc_audio = proc_audio[start_idx:]
        
        # Level normalization that PRESERVES the gain ratio between DI and processed.
        # This is critical for amp modeling: the model must learn the gain.
        # We normalize the processed (target) to a target level, then apply the same
        # scale to the DI input. This preserves the relative gain difference.
        proc_peak = np.max(np.abs(proc_audio))
        di_peak = np.max(np.abs(di_audio))
        
        if proc_peak > 0:
            # Normalize processed to target level
            target_level = 0.9
            proc_scale = target_level / proc_peak
            proc_audio = proc_audio * proc_scale
            # Apply SAME scale to DI (preserves gain ratio)
            di_audio = di_audio * proc_scale
        elif di_peak > 0:
            # If processed is silent, just normalize DI
            target_level = 0.3
            di_audio = di_audio * (target_level / di_peak)
        
        # Store as tensors
        self.di = torch.tensor(di_audio, dtype=torch.float32)
        self.processed = torch.tensor(proc_audio, dtype=torch.float32)
        
        # Calculate number of segments with half-stride overlap
        # This gives 2x more training data
        stride = self.segment_length // 2
        self.num_segments = max(0, (len(self.di) - self.segment_length) // stride)
        
        print(f"  Dataset: {len(self.di)} samples ({len(self.di)/sample_rate:.1f}s), "
              f"{self.num_segments} segments of {self.segment_length} samples "
              f"({self.segment_length/sample_rate*1000:.0f}ms)")
    
    def __len__(self) -> int:
        return self.num_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Half-stride overlap for more training data
        start = idx * (self.segment_length // 2)
        end = start + self.segment_length
        
        x = self.di[start:end]
        y = self.processed[start:end]
        
        # Pad if needed (last segment might be shorter)
        if len(x) < self.segment_length:
            x = torch.nn.functional.pad(x, (0, self.segment_length - len(x)))
            y = torch.nn.functional.pad(y, (0, self.segment_length - len(y)))
        
        return x.unsqueeze(0), y.unsqueeze(0)  # (1, T) each


class ReferenceDIProvider:
    """
    Provides a standard reference DI signal for training tone models
    when we only have the processed (target) audio.
    
    NOTE: This is now superseded by the improved DI estimator.
    Kept for backward compatibility.
    """
    
    def __init__(self, sample_rate: int = 48000, duration_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self._cache = None
    
    def generate(self) -> np.ndarray:
        """Generate a reference DI signal."""
        if self._cache is not None:
            return self._cache
        
        sr = self.sample_rate
        total_samples = int(sr * self.duration_seconds)
        signal = np.zeros(total_samples, dtype=np.float32)
        
        current_pos = 0
        
        # 1. Clean single notes (sweep across fretboard)
        for note_idx in range(12):
            freq = 82.41 * (2 ** (note_idx / 12.0))
            note_len = int(sr * 0.8)
            if current_pos + note_len > total_samples:
                break
            t = np.linspace(0, 0.8, note_len, dtype=np.float32)
            note = np.zeros(note_len, dtype=np.float32)
            for harmonic in range(1, 6):
                amplitude = 1.0 / (harmonic ** 1.5)
                note += amplitude * np.sin(2 * np.pi * freq * harmonic * t)
            envelope = np.exp(-3.0 * t)
            note = note * envelope / np.max(np.abs(note) + 1e-8) * 0.6
            signal[current_pos:current_pos + note_len] += note
            current_pos += note_len
        
        # 2. Power chords (root + fifth)
        for chord_idx in range(6):
            root_freq = 82.41 * (2 ** (chord_idx * 2 / 12.0))
            fifth_freq = root_freq * 1.5
            chord_len = int(sr * 1.0)
            if current_pos + chord_len > total_samples:
                break
            t = np.linspace(0, 1.0, chord_len, dtype=np.float32)
            chord = np.zeros(chord_len, dtype=np.float32)
            for freq in [root_freq, fifth_freq]:
                for harmonic in range(1, 4):
                    amplitude = 1.0 / (harmonic ** 1.5)
                    chord += amplitude * np.sin(2 * np.pi * freq * harmonic * t)
            envelope = np.exp(-2.0 * t)
            chord = chord * envelope / np.max(np.abs(chord) + 1e-8) * 0.7
            signal[current_pos:current_pos + chord_len] += chord
            current_pos += chord_len
        
        # 3. Palm mute style (short, percussive)
        for mute_idx in range(8):
            freq = 82.41 * (2 ** (mute_idx / 12.0))
            mute_len = int(sr * 0.15)
            if current_pos + mute_len > total_samples:
                break
            t = np.linspace(0, 0.15, mute_len, dtype=np.float32)
            note = np.zeros(mute_len, dtype=np.float32)
            for harmonic in range(1, 8):
                amplitude = 1.0 / (harmonic ** 1.0)
                note += amplitude * np.sin(2 * np.pi * freq * harmonic * t)
            envelope = np.exp(-15.0 * t)
            note = note * envelope / np.max(np.abs(note) + 1e-8) * 0.8
            signal[current_pos:current_pos + mute_len] += note
            current_pos += int(sr * 0.1)
        
        # 4. Legato / slides
        slide_len = int(sr * 2.0)
        if current_pos + slide_len <= total_samples:
            t = np.linspace(0, 2.0, slide_len, dtype=np.float32)
            start_freq = 110.0
            end_freq = 440.0
            freq_slide = start_freq * np.exp(t * np.log(end_freq / start_freq) / 2.0)
            slide = np.zeros(slide_len, dtype=np.float32)
            for harmonic in range(1, 4):
                phase = 2 * np.pi * harmonic * np.cumsum(freq_slide) / sr
                amplitude = 1.0 / (harmonic ** 1.5)
                slide += amplitude * np.sin(phase)
            envelope = np.exp(-1.5 * t)
            slide = slide * envelope / np.max(np.abs(slide) + 1e-8) * 0.5
            signal[current_pos:current_pos + slide_len] += slide
            current_pos += slide_len
        
        # 5. Noise floor
        noise = np.random.randn(total_samples).astype(np.float32) * 0.002
        signal += noise
        
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.8
        
        self._cache = signal
        return signal
    
    def save(self, path: str):
        """Save reference DI to WAV file."""
        signal = self.generate()
        sf.write(path, signal, self.sample_rate)