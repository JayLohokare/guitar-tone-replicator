"""
Tone Replicator - Local API Server
FastAPI backend with full end-to-end pipeline:
  Video URL → Download → Guitar Stem → Train Tone Model → CoreML Export
"""
import os
import sys
import json
import uuid
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.trainer import ToneTrainer
from src.separation.separator import GuitarSeparator, prepare_target_tone

app = FastAPI(
    title="Tone Replicator API",
    description="Local API for guitar tone replication — end-to-end pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Storage
MODELS_DIR = Path.home() / "ToneReplicator" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path(tempfile.gettempdir()) / "tone_replicator_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path(tempfile.gettempdir()) / "tone_replicator_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Active training jobs
jobs: Dict[str, dict] = {}

# ============================================================
# Models
# ============================================================

class TrainRequest(BaseModel):
    song_path: str = ""
    model_name: str = "MyTone"
    model_type: str = "wavenet"
    model_size: str = "standard"
    epochs: int = 100
    skip_separation: bool = False

class ApplyRequest(BaseModel):
    model_path: str
    di_path: str
    output_path: str = ""

class PipelineRequest(BaseModel):
    """End-to-end: URL → train → CoreML model ready."""
    url: str
    model_name: str = "MyTone"
    model_type: str = "wavenet"
    model_size: str = "standard"  # v2: standard default (bigger receptive field)
    epochs: int = 200  # v2: more epochs for better convergence
    start_time: Optional[float] = None   # seconds into the video
    end_time: Optional[float] = None     # seconds into the video
    convert_coreml: bool = True

class URLDownloadRequest(BaseModel):
    url: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

# ============================================================
# End-to-End Pipeline (the main flow)
# ============================================================

@app.post("/pipeline")
async def start_pipeline(request: PipelineRequest):
    """
    End-to-end pipeline:
    1. Download audio from URL (YouTube, direct link, etc.)
    2. Extract guitar stem
    3. Train tone model
    4. (Optional) Convert to CoreML for AUv3 plugin
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "starting",
        "step": "starting",
        "progress": 0,
        "message": "Initializing pipeline...",
        "model_path": None,
        "coreml_path": None,
        "result": None,
    }
    
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()
    
    return {"job_id": job_id}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

# ============================================================
# Download from URL
# ============================================================

@app.post("/download_url")
async def download_from_url(request: URLDownloadRequest):
    """Download audio from a URL (YouTube, SoundCloud, direct link)."""
    try:
        output_path = _download_audio(request.url, request.start_time, request.end_time)
        return {"path": str(output_path), "filename": Path(output_path).name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Train (standalone)
# ============================================================

@app.post("/train")
async def start_training(request: TrainRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "starting", "step": "starting", "progress": 0,
        "message": "Initializing...", "model_path": None, "result": None,
    }
    thread = threading.Thread(target=_run_training, args=(job_id, request), daemon=True)
    thread.start()
    return {"job_id": job_id}

# ============================================================
# Apply
# ============================================================

@app.post("/apply")
async def apply_tone(request: ApplyRequest):
    trainer = ToneTrainer(device="auto")
    trainer.load_model(request.model_path)
    output_path = request.output_path or str(Path(tempfile.gettempdir()) / "tone_applied.wav")
    result_path = trainer.process_audio(request.di_path, output_path)
    return {"output_path": result_path, "status": "ok"}

# ============================================================
# Upload
# ============================================================

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = UPLOADS_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"path": str(file_path), "filename": file.filename}

# ============================================================
# Models
# ============================================================

@app.get("/models")
async def list_models():
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and (model_dir / "metadata.json").exists():
            with open(model_dir / "metadata.json") as f:
                metadata = json.load(f)
            coreml_exists = (model_dir / "model.mlpackage").exists()
            models.append({
                "id": model_dir.name,
                "name": model_dir.name,
                "model_path": str(model_dir),
                "coreml_ready": coreml_exists,
                **metadata,
            })
    return models

@app.post("/models/{model_id}/convert_coreml")
async def convert_to_coreml(model_id: str):
    """Convert a trained PyTorch model to CoreML format for the AUv3 plugin."""
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        coreml_path = _convert_to_coreml(str(model_dir))
        return {"status": "ok", "coreml_path": str(coreml_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Internal: Pipeline Runner
# ============================================================

def _run_pipeline(job_id: str, request: PipelineRequest):
    """Run the full pipeline in a background thread."""
    try:
        # Step 1: Download audio from URL
        jobs[job_id].update({
            "status": "downloading", "step": "downloading",
            "progress": 5, "message": f"Downloading from URL...",
        })
        
        audio_path = _download_audio(request.url, request.start_time, request.end_time)
        jobs[job_id].update({"progress": 15, "message": f"Downloaded: {Path(audio_path).name}"})
        
        # Step 2: Extract guitar stem
        jobs[job_id].update({
            "status": "separating", "step": "separating",
            "progress": 20, "message": "Separating guitar stem...",
        })
        
        separator = GuitarSeparator()
        if separator.health_check():
            guitar_stem = separator.separate_file(audio_path)
            jobs[job_id].update({"progress": 35, "message": "Guitar stem extracted!"})
        else:
            # If separation API not available, use the whole file
            guitar_stem = audio_path
            jobs[job_id].update({"progress": 35, "message": "Using full audio (separation API not available)"})
        
        # Step 3: Prepare target tone
        jobs[job_id].update({
            "status": "preparing", "step": "preparing",
            "progress": 40, "message": "Preparing training data...",
        })
        prepared_stem = prepare_target_tone(guitar_stem)
        
        # Step 4: Train the model
        jobs[job_id].update({
            "status": "training", "step": "training",
            "progress": 45, "message": "Training tone model...",
        })
        
        model_dir = MODELS_DIR / request.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = ToneTrainer(
            model_type=request.model_type,
            model_size=request.model_size,
            segment_length=16384,  # v2: longer segments
            pre_emphasis=0.85,     # v2: pre-emphasis filter
            mrstft_weight=0.25,    # v2: more perceptual loss weight
        )
        
        def progress_callback(entry):
            epoch = entry.get("epoch", 0)
            val_esr = entry.get("val_esr", 1.0)
            pct = 45 + min(epoch / request.epochs, 1.0) * 45
            jobs[job_id].update({
                "epoch": epoch,
                "val_esr": val_esr,
                "progress": pct,
                "message": f"Epoch {epoch}: ESR = {val_esr:.4f}",
            })
        
        result = trainer.train_blind(
            target_tone_path=prepared_stem,
            epochs=request.epochs,
            save_path=str(model_dir),
            progress_callback=progress_callback,
        )
        
        # Step 5: Convert to CoreML
        coreml_path = None
        if request.convert_coreml:
            jobs[job_id].update({
                "status": "converting", "step": "converting",
                "progress": 92, "message": "Converting to CoreML...",
            })
            try:
                coreml_path = _convert_to_coreml(str(model_dir))
                jobs[job_id].update({"message": "CoreML model ready!"})
            except Exception as e:
                # CoreML conversion requires Python 3.12 (coremltools binary compat)
                # Provide helpful message
                import sys
                msg = f"CoreML conversion requires Python 3.12 (running {sys.version_info.major}.{sys.version_info.minor}). "
                msg += "Run: python3.12 CoreMLConverter/convert_model.py --model-dir " + str(model_dir)
                jobs[job_id].update({"message": f"PyTorch model saved. {msg}"})
        
        # Done!
        jobs[job_id].update({
            "status": "complete", "step": "complete",
            "progress": 100,
            "message": f"Done! ESR = {result['best_val_esr']:.4f}",
            "model_path": str(model_dir),
            "coreml_path": str(coreml_path) if coreml_path else None,
            "result": result,
        })
        
    except Exception as e:
        import traceback
        jobs[job_id].update({
            "status": "error", "step": "error",
            "message": f"{type(e).__name__}: {str(e)}",
        })

def _run_training(job_id: str, request: TrainRequest):
    """Run training only (legacy endpoint)."""
    try:
        # Separation
        jobs[job_id].update({
            "status": "separating", "step": "separating",
            "progress": 0, "message": "Separating guitar stem...",
        })
        
        if request.skip_separation:
            guitar_stem = request.song_path
        else:
            separator = GuitarSeparator()
            guitar_stem = separator.separate_file(request.song_path) if separator.health_check() else request.song_path
        
        jobs[job_id].update({"progress": 20, "message": "Guitar stem ready!"})
        
        # Prepare
        jobs[job_id].update({
            "status": "preparing", "step": "preparing",
            "progress": 25, "message": "Preparing training data...",
        })
        prepared_stem = prepare_target_tone(guitar_stem)
        jobs[job_id].update({"progress": 30, "message": "Training data prepared!"})
        
        # Train
        model_dir = MODELS_DIR / request.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = ToneTrainer(
            model_type=request.model_type, 
            model_size=request.model_size,
            segment_length=16384,  # v2: longer segments
            pre_emphasis=0.85,     # v2: pre-emphasis
            mrstft_weight=0.25,    # v2: more perceptual loss
        )
        
        def progress_callback(entry):
            epoch = entry.get("epoch", 0)
            val_esr = entry.get("val_esr", 1.0)
            pct = 30 + min(epoch / request.epochs, 1.0) * 65
            jobs[job_id].update({
                "status": "training", "step": "training",
                "epoch": epoch, "val_esr": val_esr, "progress": pct,
                "message": f"Epoch {epoch}: ESR = {val_esr:.4f}",
            })
        
        result = trainer.train_blind(
            target_tone_path=prepared_stem,
            epochs=request.epochs,
            save_path=str(model_dir),
            progress_callback=progress_callback,
        )
        
        jobs[job_id].update({
            "status": "complete", "step": "complete", "progress": 100,
            "message": f"Training complete! ESR = {result['best_val_esr']:.4f}",
            "model_path": str(model_dir), "result": result,
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "error", "step": "error", "message": str(e),
        })

# ============================================================
# Internal: Audio Download
# ============================================================

def _download_audio(url: str, start_time: float = None, end_time: float = None) -> str:
    """
    Download audio from a URL. Supports:
    - YouTube (via yt-dlp)
    - Direct links to audio/video files
    - Local file paths (file:// or absolute paths)
    - SoundCloud, etc.
    """
    # Handle local files
    if url.startswith("file://"):
        local_path = url.replace("file://", "")
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    if os.path.exists(url):
        return url
    
    # Check if it's a direct audio/video link
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff'}
    url_path = Path(url.split('?')[0].split('#')[0])
    
    if url_path.suffix.lower() in audio_extensions:
        import requests
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        ext = url_path.suffix.lower()
        output_path = DOWNLOADS_DIR / f"download_{uuid.uuid4().hex[:8]}{ext}"
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(output_path)
    
    # Use yt-dlp for video sites (YouTube, SoundCloud, etc.)
    output_template = str(DOWNLOADS_DIR / "download_%(id)s.%(ext)s")
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--quiet",
    ]
    
    cmd.append(url)
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    # Find the downloaded file
    downloaded = sorted(DOWNLOADS_DIR.glob("download_*.*"), key=lambda p: p.stat().st_mtime)
    if not downloaded:
        raise RuntimeError("No audio file downloaded")
    
    audio_path = str(downloaded[-1])
    
    # Trim if time range specified
    if start_time is not None or end_time is not None:
        trimmed_path = str(DOWNLOADS_DIR / f"trimmed_{Path(audio_path).stem}.wav")
        trim_cmd = ["ffmpeg", "-y", "-i", audio_path]
        if start_time is not None:
            trim_cmd.extend(["-ss", str(start_time)])
        if end_time is not None:
            trim_cmd.extend(["-to", str(end_time)])
        trim_cmd.extend(["-ar", "44100", "-ac", "1", trimmed_path])
        
        result = subprocess.run(trim_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            audio_path = trimmed_path
    
    return audio_path

# ============================================================
# Internal: CoreML Conversion
# ============================================================

def _convert_to_coreml(model_dir: str) -> Path:
    """Convert a trained PyTorch model to CoreML format."""
    import torch
    import coremltools as ct
    from src.core.model import create_model
    
    model_dir = Path(model_dir)
    checkpoint = torch.load(
        model_dir / "model.pth",
        map_location="cpu",
        weights_only=True,
    )
    
    model = create_model(checkpoint["model_type"], checkpoint["model_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Trace with sample input
    sample_rate = checkpoint.get("sample_rate", 44100)
    sample_input = torch.randn(1, 1, sample_rate * 2)  # 2 seconds of audio
    
    traced = torch.jit.trace(model, sample_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=sample_input.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    
    # Save
    coreml_path = model_dir / "model.mlpackage"
    mlmodel.save(str(coreml_path))
    
    # Update metadata
    metadata = {}
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata["coreml_path"] = str(coreml_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return coreml_path

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("🎸 Tone Replicator API v2.0 starting on http://localhost:8767")
    print("   Endpoints: /pipeline (URL→tone), /train, /apply, /models")
    uvicorn.run(app, host="0.0.0.0", port=8767)