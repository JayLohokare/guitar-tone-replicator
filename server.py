"""
Guitar Separation API v2 - Apple Silicon Optimized
Uses Demucs-MLX for faster inference on Apple Silicon
"""

import os
import uuid
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Guitar Separation API v2",
    description="Apple Silicon optimized guitar stem extraction using Demucs-MLX",
    version="2.0.0"
)

# Directories
UPLOAD_DIR = Path(os.path.expanduser("~/guitar-api-v2/uploads"))
OUTPUT_DIR = Path(os.path.expanduser("~/guitar-api-v2/outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job tracking
jobs = {}

# Model configuration
MODEL_NAME = "htdemucs_6s"  # 6-source model with guitar stem
SAMPLE_RATE = 44100
MAX_FILE_SIZE_MB = 100
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

# Global model holder
separator = None
USE_MLX = False


def get_separator():
    """Lazy load the separator model"""
    global separator, USE_MLX
    if separator is None:
        logger.info("Loading Demucs model...")
        try:
            # Try MLX first (Apple Silicon optimized)
            from demucs_mlx import Separator
            logger.info("Using Demucs-MLX (Apple Silicon optimized)")
            separator = Separator(model_name=MODEL_NAME)
            USE_MLX = True
            logger.info("Model loaded successfully (MLX)")
        except Exception as e:
            logger.warning(f"MLX not available: {e}, falling back to PyTorch")
            # Fallback to regular demucs
            import torch
            from demucs.pretrained import get_model
            logger.info("Using standard Demucs (PyTorch)")
            separator = get_model(MODEL_NAME)
            USE_MLX = False
            logger.info("Model loaded successfully (PyTorch)")
    return separator


def separate_audio(input_path: str, output_path: str, job_id: str) -> dict:
    """Separate audio and extract guitar stem"""
    import time
    
    separator = get_separator()
    
    logger.info(f"[{job_id}] Processing {input_path}")
    start_time = time.time()
    
    if USE_MLX:
        # MLX path - Apple Silicon optimized
        # separate_audio_file returns tuple: (full_audio, sources_dict)
        result = separator.separate_audio_file(input_path)
        
        # Unpack the result
        full_audio, sources = result
        
        # Sources dict has keys: drums, bass, other, vocals, guitar, piano
        guitar_audio = sources['guitar']
        
        logger.info(f"[{job_id}] Guitar stem extracted: shape={guitar_audio.shape}")
        
        # Save as WAV using soundfile (MLX doesn't have native audio save)
        import soundfile as sf
        sf.write(output_path, guitar_audio.T, SAMPLE_RATE)  # Transpose for (samples, channels)
        
    else:
        # Standard Demucs PyTorch path
        import torch
        import torchaudio
        from demucs.apply import apply_model
        
        # Load audio
        waveform, sr = torchaudio.load(input_path)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(separator, waveform[None], progress=False)[0]
        
        # Sources order for htdemucs_6s: drums, bass, other, vocals, guitar, piano
        # guitar is at index 4
        guitar_wave = sources[4]
        
        # Save as WAV
        torchaudio.save(output_path, guitar_wave, SAMPLE_RATE)
    
    elapsed = time.time() - start_time
    
    # Get file sizes
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    
    logger.info(f"[{job_id}] Completed in {elapsed:.1f}s")
    
    return {
        "status": "completed",
        "input_size_mb": round(input_size / (1024 * 1024), 2),
        "output_size_mb": round(output_size / (1024 * 1024), 2),
        "output_file": output_path,
        "engine": "MLX" if USE_MLX else "PyTorch",
        "processing_time_seconds": round(elapsed, 1)
    }


@app.get("/")
async def health_check():
    """Health check endpoint"""
    global USE_MLX
    return {
        "status": "online",
        "service": "Guitar Separation API v2",
        "model": MODEL_NAME,
        "optimization": "Apple Silicon (MLX)" if USE_MLX else "PyTorch",
        "max_file_size_mb": MAX_FILE_SIZE_MB
    }


@app.post("/separate")
async def separate_endpoint(file: UploadFile = File(...)):
    """Upload audio file for guitar extraction"""
    
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    
    try:
        # Read and validate size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
            )
        
        with open(input_path, 'wb') as f:
            f.write(content)
        
        # Track job
        jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "started": asyncio.get_event_loop().time()
        }
        
        # Output path
        output_path = OUTPUT_DIR / f"{job_id}_guitar.wav"
        
        # Run separation in thread pool
        result = await run_in_threadpool(
            separate_audio,
            str(input_path),
            str(output_path),
            job_id
        )
        
        jobs[job_id].update(result)
        
        return {"job_id": job_id, "status": "processing"}
        
    except HTTPException:
        raise
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    response = {"job_id": job_id, "status": job["status"]}
    
    if "error" in job:
        response["error"] = job["error"]
    
    if "output_size_mb" in job:
        response["output_size_mb"] = job["output_size_mb"]
    
    return response


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download extracted guitar stem"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}"
        )
    
    output_path = OUTPUT_DIR / f"{job_id}_guitar.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{Path(job['filename']).stem}_guitar.wav"
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Guitar Separation API v2")
    parser.add_argument("--port", type=int, default=8766, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    logger.info(f"Starting Guitar Separation API v2 on {args.host}:{args.port}")
    logger.info("Apple Silicon optimization: MLX enabled when available")
    uvicorn.run(app, host=args.host, port=args.port)