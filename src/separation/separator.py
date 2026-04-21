"""
Tone Replicator - Guitar Separation Bridge
Connects to the local Demucs-MLX v2 API for guitar stem extraction.
"""

import os
import time
import requests
import tempfile
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional


class GuitarSeparator:
    """Interface to the guitar separation API on the Mac mini."""
    
    # Default v2 API endpoint
    DEFAULT_API_URL = "http://localhost:8766"
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or self.DEFAULT_API_URL
    
    def health_check(self) -> bool:
        """Check if the separation API is running."""
        try:
            resp = requests.get(f"{self.api_url}/", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
    
    def separate_file(self, audio_path: str) -> str:
        """
        Upload an audio file and extract the guitar stem.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path to the extracted guitar stem WAV file
        """
        # Submit job
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/separate",
                files={"file": f},
            )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        
        # Poll for completion
        while True:
            resp = requests.get(f"{self.api_url}/status/{job_id}")
            resp.raise_for_status()
            status = resp.json()
            
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                raise RuntimeError(f"Separation failed: {status.get('error', 'unknown')}")
            
            time.sleep(1)
        
        # Download result
        resp = requests.get(f"{self.api_url}/download/{job_id}")
        resp.raise_for_status()
        
        # Save to temp file
        output_dir = Path(audio_path).parent / "separated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"guitar_stem_{Path(audio_path).stem}.wav")
        
        with open(output_path, "wb") as f:
            f.write(resp.content)
        
        return output_path
    
    def separate_url(self, url: str, output_dir: str = "/tmp") -> str:
        """
        Download audio from URL, then separate guitar stem.
        
        Args:
            url: URL to audio file (YouTube, direct link, etc.)
            output_dir: Directory for downloaded files
            
        Returns:
            Path to the extracted guitar stem WAV file
        """
        # For now, expect a direct audio file URL
        # YouTube download would be handled separately
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        
        # Determine extension
        ext = Path(url).suffix or ".wav"
        temp_path = os.path.join(output_dir, f"downloaded_audio{ext}")
        
        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return self.separate_file(temp_path)


def prepare_target_tone(
    audio_path: str,
    target_sr: int = 48000,
    trim_silence: bool = True,
    normalize: bool = True,
    max_duration: float = 60.0,
) -> str:
    """
    Prepare an audio file for use as a training target.
    
    - Resamples to target sample rate
    - Converts to mono
    - Trims silence
    - Normalizes
    - Limits duration
    
    Returns path to prepared audio file.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=max_duration)
    
    # Trim silence
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    
    # Normalize
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
    
    # Save prepared audio
    output_path = str(Path(audio_path).parent / f"{Path(audio_path).stem}_prepared.wav")
    sf.write(output_path, audio, target_sr)
    
    return output_path