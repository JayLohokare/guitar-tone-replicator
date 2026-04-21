"""
Tone Replicator - Main Pipeline
End-to-end: Song → Guitar Stem → Train Tone Model → Apply to new audio
"""

import os
import json
import time
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from ..core.model import create_model
from ..core.trainer import ToneTrainer
from ..core.dataset import ReferenceDIProvider
from ..separation.separator import GuitarSeparator, prepare_target_tone


class ToneReplicatorPipeline:
    """
    Complete pipeline for guitar tone replication.
    
    Usage:
        pipeline = ToneReplicatorPipeline()
        
        # From a song URL or file
        result = pipeline.replicate_from_song(
            song_path="song.mp3",
            output_dir="output",
        )
        
        # From a guitar stem (already separated)
        result = pipeline.replicate_from_stem(
            stem_path="guitar_stem.wav",
            output_dir="output",
        )
        
        # Apply learned tone to your own DI
        output = pipeline.apply_tone(
            model_path="output/model",
            di_path="my_recording.wav",
            output_path="output/my_tone.wav",
        )
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8766",
        sample_rate: int = 48000,
        model_type: str = "wavenet",
        model_size: str = "standard",
        device: str = "auto",
        epochs: int = 100,
    ):
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.model_size = model_size
        self.epochs = epochs
        self.separator = GuitarSeparator(api_url=api_url)
        self.device = device
    
    def replicate_from_song(
        self,
        song_path: str,
        output_dir: str = "./output",
        progress_callback: Optional[Callable] = None,
        skip_separation: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: Song → Guitar Stem → Train Model
        
        Args:
            song_path: Path to audio file or URL
            output_dir: Where to save outputs
            progress_callback: Callback for progress updates
            skip_separation: If True, assume song_path is already a guitar stem
            
        Returns:
            Dict with model path, training results, etc.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Separate guitar stem
        if not skip_separation:
            if progress_callback:
                progress_callback({"step": "separating", "progress": 0})
            guitar_stem = self.separator.separate_file(song_path)
            if progress_callback:
                progress_callback({"step": "separating", "progress": 100})
        else:
            guitar_stem = song_path
        
        # Step 2: Prepare target tone
        if progress_callback:
            progress_callback({"step": "preparing", "progress": 0})
        prepared_stem = prepare_target_tone(
            guitar_stem,
            target_sr=self.sample_rate,
            max_duration=60.0,  # Use first 60 seconds
        )
        if progress_callback:
            progress_callback({"step": "preparing", "progress": 100})
        
        # Step 3: Train tone model
        if progress_callback:
            progress_callback({"step": "training", "progress": 0})
        
        trainer = ToneTrainer(
            model_type=self.model_type,
            model_size=self.model_size,
            sample_rate=self.sample_rate,
            device=self.device,
        )
        
        result = trainer.train_blind(
            target_tone_path=prepared_stem,
            epochs=self.epochs,
            save_path=str(output_dir / "model"),
            progress_callback=lambda entry: progress_callback({
                "step": "training",
                "epoch": entry["epoch"],
                "train_esr": entry["train_esr"],
                "val_esr": entry["val_esr"],
            }) if progress_callback else None,
        )
        
        if progress_callback:
            progress_callback({"step": "training", "progress": 100})
        
        # Step 4: Save summary
        summary = {
            "model_path": str(output_dir / "model"),
            "guitar_stem_path": guitar_stem,
            "prepared_stem_path": prepared_stem,
            "best_val_esr": result["best_val_esr"],
            "model_type": self.model_type,
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "epochs_trained": len(result["history"]),
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def replicate_from_stem(
        self,
        stem_path: str,
        output_dir: str = "./output",
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train from an already-extracted guitar stem.
        """
        return self.replicate_from_song(
            song_path=stem_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
            skip_separation=True,
        )
    
    def apply_tone(
        self,
        model_path: str,
        di_path: str,
        output_path: str,
    ) -> str:
        """
        Apply a trained tone model to a DI recording.
        
        Args:
            model_path: Path to saved model directory
            di_path: Path to clean DI recording
            output_path: Where to save the tone-processed output
            
        Returns:
            Path to the output file
        """
        trainer = ToneTrainer(
            model_type=self.model_type,
            model_size=self.model_size,
            sample_rate=self.sample_rate,
            device=self.device,
        )
        trainer.load_model(model_path)
        
        return trainer.process_audio(di_path, output_path)


# Convenience function for one-shot tone replication
def replicate_tone(
    song_path: str,
    di_path: str,
    output_path: str = "output/tone_applied.wav",
    model_type: str = "wavenet",
    model_size: str = "standard",
    epochs: int = 100,
) -> str:
    """
    One-shot: Song → Extract tone → Apply to your DI → Save output.
    """
    pipeline = ToneReplicatorPipeline(
        model_type=model_type,
        model_size=model_size,
        epochs=epochs,
    )
    
    # Train from song
    result = pipeline.replicate_from_song(
        song_path=song_path,
        output_dir=str(Path(output_path).parent / "model"),
    )
    
    # Apply to DI
    output = pipeline.apply_tone(
        model_path=result["model_path"],
        di_path=di_path,
        output_path=output_path,
    )
    
    return output