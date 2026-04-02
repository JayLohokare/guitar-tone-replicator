"""
Guitar Separation API v2 - Apple Silicon Optimized
Uses Demucs-MLX for faster inference on Apple Silicon
"""

import os
import uuid
import asyncio
import logging
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Guitar Separation API v2",
    description="Apple Silicon optimized guitar stem extraction using Demucs-MLX",
    version="2.1.0"
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
SUPPORTED_INPUT_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
SUPPORTED_OUTPUT_FORMATS = {'wav', 'mp3', 'ogg', 'flac'}

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


def convert_audio(input_path: str, output_path: str, output_format: str) -> str:
    """Convert audio to desired format using ffmpeg"""
    output_format = output_format.lower()
    
    if output_format == 'wav':
        # Already WAV, just copy
        if input_path.endswith('.wav'):
            import shutil
            shutil.copy(input_path, output_path)
            return output_path
        # Convert to WAV
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'pcm_s16le', output_path]
    elif output_format == 'mp3':
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'libmp3lame', '-q:a', '0', output_path]
    elif output_format == 'ogg':
        # OGG/Opus - good compression, great for messaging
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'libopus', '-b:a', '128k', output_path]
    elif output_format == 'flac':
        # FLAC - lossless compression
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'flac', output_path]
    else:
        # Default to WAV
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'pcm_s16le', output_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise Exception(f"Audio conversion failed: {result.stderr[:200]}")
    
    return output_path


def separate_audio(input_path: str, output_path: str, job_id: str, output_format: str = 'wav') -> dict:
    """Separate audio and extract guitar stem"""
    import time
    
    separator = get_separator()
    
    logger.info(f"[{job_id}] Processing {input_path}")
    start_time = time.time()
    
    # Always extract to WAV first
    wav_output_path = output_path if output_format == 'wav' else output_path.replace(f'.{output_format}', '.wav')
    
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
        sf.write(wav_output_path, guitar_audio.T, SAMPLE_RATE)  # Transpose for (samples, channels)
        
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
        torchaudio.save(wav_output_path, guitar_wave, SAMPLE_RATE)
    
    # Convert to desired format if not WAV
    final_output_path = output_path
    if output_format != 'wav':
        logger.info(f"[{job_id}] Converting to {output_format.upper()}")
        convert_audio(wav_output_path, output_path, output_format)
        # Remove intermediate WAV
        os.remove(wav_output_path)
    
    elapsed = time.time() - start_time
    
    # Get file sizes
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(final_output_path)
    
    logger.info(f"[{job_id}] Completed in {elapsed:.1f}s ({output_format.upper()}, {output_size / (1024*1024):.1f}MB)")
    
    return {
        "status": "completed",
        "input_size_mb": round(input_size / (1024 * 1024), 2),
        "output_size_mb": round(output_size / (1024 * 1024), 2),
        "output_file": final_output_path,
        "output_format": output_format,
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
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "youtube_support": True,
        "supported_output_formats": list(SUPPORTED_OUTPUT_FORMATS),
        "default_output_format": "wav"
    }


def download_youtube_audio(url: str, output_dir: Path, job_id: str) -> tuple:
    """Download audio from YouTube URL and return (audio_path, title)"""
    import time
    
    logger.info(f"[{job_id}] Downloading YouTube: {url}")
    start_time = time.time()
    
    # Output template - use fixed extension since we convert to wav
    output_path = output_dir / f"{job_id}"
    output_template = str(output_path)
    
    # yt-dlp command - download best audio, convert to wav
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--no-playlist",  # Single video only
        "--no-warnings",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown download error"
            logger.error(f"[{job_id}] YouTube download failed: {error_msg}")
            raise Exception(f"YouTube download failed: {error_msg}")
        
        # yt-dlp converts to wav and saves as {job_id}.wav
        audio_file = output_dir / f"{job_id}.wav"
        
        if not audio_file.exists():
            # Fallback: find any file with job_id prefix
            downloaded_files = list(output_dir.glob(f"{job_id}.*"))
            if downloaded_files:
                audio_file = downloaded_files[0]
            else:
                raise Exception(f"Downloaded file not found. stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}")
        
        # Extract title from stdout (yt-dlp prints progress info)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        title = f"YouTube_{job_id}"
        # Try to find title line
        for line in reversed(lines):
            if line and not line.startswith('[') and len(line) > 3:
                title = re.sub(r'[^\w\s-]', '', line)[:100]
                break
        
        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Downloaded in {elapsed:.1f}s: {audio_file.name}")
        
        return str(audio_file), title
        
    except subprocess.TimeoutExpired:
        raise Exception("YouTube download timed out (5 min limit)")
    except FileNotFoundError:
        raise Exception("yt-dlp not installed. Run: brew install yt-dlp")


@app.post("/youtube")
async def youtube_endpoint(
    url: str = Form(...),
    format: str = Form("wav")
):
    """Download YouTube audio and extract guitar stem
    
    Parameters:
    - url: YouTube video URL
    - format: Output format (wav, mp3, ogg, flac). Default: wav
    
    Returns:
    - job_id: Use to check status and download result
    """
    
    # Validate format
    output_format = format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )
    
    # Validate YouTube URL
    youtube_patterns = [
        r'(youtube\.com/watch\?v=)',
        r'(youtu\.be/)',
        r'(youtube\.com/shorts/)',
        r'(youtube\.com/embed/)',
    ]
    
    if not any(re.search(p, url) for p in youtube_patterns):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Supported: youtube.com/watch, youtu.be, shorts"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Track job
    jobs[job_id] = {
        "status": "downloading",
        "url": url,
        "output_format": output_format,
        "started": asyncio.get_event_loop().time()
    }
    
    try:
        # Download YouTube audio
        audio_path, title = await run_in_threadpool(
            download_youtube_audio,
            url,
            UPLOAD_DIR,
            job_id
        )
        
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["title"] = title
        
        # Output path with correct extension
        output_path = OUTPUT_DIR / f"{job_id}_guitar.{output_format}"
        
        # Run separation in thread pool
        result = await run_in_threadpool(
            separate_audio,
            audio_path,
            str(output_path),
            job_id,
            output_format
        )
        
        jobs[job_id].update(result)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "title": title,
            "output_format": output_format
        }
        
    except HTTPException:
        raise
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/separate")
async def separate_endpoint(
    file: UploadFile = File(...),
    format: str = Form("wav")
):
    """Upload audio file for guitar extraction
    
    Parameters:
    - file: Audio file (mp3, wav, flac, ogg, m4a, aac, wma)
    - format: Output format (wav, mp3, ogg, flac). Default: wav
    
    Returns:
    - job_id: Use to check status and download result
    """
    
    # Validate format
    output_format = format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )
    
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_INPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_INPUT_FORMATS)}"
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
            "output_format": output_format,
            "started": asyncio.get_event_loop().time()
        }
        
        # Output path with correct extension
        output_path = OUTPUT_DIR / f"{job_id}_guitar.{output_format}"
        
        # Run separation in thread pool
        result = await run_in_threadpool(
            separate_audio,
            str(input_path),
            str(output_path),
            job_id,
            output_format
        )
        
        jobs[job_id].update(result)
        
        return {"job_id": job_id, "status": "processing", "output_format": output_format}
        
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
    
    if "output_format" in job:
        response["output_format"] = job["output_format"]
    
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
    
    output_format = job.get("output_format", "wav")
    output_path = OUTPUT_DIR / f"{job_id}_guitar.{output_format}"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Determine media type
    media_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'ogg': 'audio/ogg',
        'flac': 'audio/flac'
    }
    
    filename_base = job.get("filename", "audio")
    if "title" in job:
        filename_base = job["title"]
    
    return FileResponse(
        output_path,
        media_type=media_types.get(output_format, "audio/wav"),
        filename=f"{Path(filename_base).stem}_guitar.{output_format}"
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