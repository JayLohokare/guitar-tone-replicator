"""
Stem Separation API v3 - Apple Silicon Optimized
Uses Demucs-MLX for faster inference on Apple Silicon
Extracts all stems: vocals, drums, bass, other, guitar, piano
"""

import os
import uuid
import asyncio
import logging
import subprocess
import re
import zipfile
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stem Separation API v3",
    description="Apple Silicon optimized stem extraction using Demucs-MLX. Extracts all stems: vocals, drums, bass, other, guitar, piano",
    version="3.0.0"
)

# Enable CORS for GitHub Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jaylohokare.github.io",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*",  # Allow all origins for flexibility
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path(os.path.expanduser("~/guitar-api-v2/uploads"))
OUTPUT_DIR = Path(os.path.expanduser("~/guitar-api-v2/outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job tracking
jobs = {}

# Model configuration
MODEL_NAME = "htdemucs_6s"  # 6-source model
SAMPLE_RATE = 44100
MAX_FILE_SIZE_MB = 100
SUPPORTED_INPUT_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
SUPPORTED_OUTPUT_FORMATS = {'wav', 'mp3', 'ogg', 'flac'}

# All available stems from htdemucs_6s
STEMS = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
STEM_ORDER = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']  # PyTorch order

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
        if input_path.endswith('.wav'):
            import shutil
            shutil.copy(input_path, output_path)
            return output_path
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'pcm_s16le', output_path]
    elif output_format == 'mp3':
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'libmp3lame', '-q:a', '0', output_path]
    elif output_format == 'ogg':
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'libopus', '-b:a', '128k', output_path]
    elif output_format == 'flac':
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'flac', output_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'pcm_s16le', output_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg conversion failed: {result.stderr}")
        raise Exception(f"Audio conversion failed: {result.stderr[:200]}")
    
    return output_path


def separate_audio_all(
    input_path: str,
    output_dir: Path,
    job_id: str,
    output_format: str = 'wav',
    stems: Optional[List[str]] = None
) -> dict:
    """Separate audio and extract all stems"""
    import time
    
    separator = get_separator()
    stems_to_extract = stems if stems else STEMS
    
    logger.info(f"[{job_id}] Processing {input_path} for stems: {stems_to_extract}")
    start_time = time.time()
    
    extracted_stems = {}
    
    if USE_MLX:
        # MLX path - Apple Silicon optimized
        result = separator.separate_audio_file(input_path)
        full_audio, sources = result
        
        logger.info(f"[{job_id}] Separation complete, saving stems...")
        
        import soundfile as sf
        
        for stem_name in stems_to_extract:
            if stem_name in sources:
                stem_audio = sources[stem_name]
                wav_path = output_dir / f"{job_id}_{stem_name}.wav"
                sf.write(str(wav_path), stem_audio.T, SAMPLE_RATE)
                extracted_stems[stem_name] = str(wav_path)
                logger.info(f"[{job_id}] Saved {stem_name}: shape={stem_audio.shape}")
        
    else:
        # Standard Demucs PyTorch path
        import torch
        import torchaudio
        from demucs.apply import apply_model
        
        waveform, sr = torchaudio.load(input_path)
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        with torch.no_grad():
            sources = apply_model(separator, waveform[None], progress=False)[0]
        
        # Sources order: drums, bass, other, vocals, guitar, piano
        for i, stem_name in enumerate(STEM_ORDER):
            if stem_name in stems_to_extract:
                stem_wave = sources[i]
                wav_path = output_dir / f"{job_id}_{stem_name}.wav"
                torchaudio.save(str(wav_path), stem_wave, SAMPLE_RATE)
                extracted_stems[stem_name] = str(wav_path)
                logger.info(f"[{job_id}] Saved {stem_name}")
    
    # Convert to desired format if not WAV
    final_stems = {}
    input_size = os.path.getsize(input_path)
    
    for stem_name, wav_path in extracted_stems.items():
        if output_format == 'wav':
            final_path = wav_path
        else:
            final_path = wav_path.replace('.wav', f'.{output_format}')
            convert_audio(wav_path, final_path, output_format)
            os.remove(wav_path)
        
        final_stems[stem_name] = {
            "path": final_path,
            "size_mb": round(os.path.getsize(final_path) / (1024 * 1024), 2)
        }
    
    # Create zip file with all stems
    zip_path = output_dir / f"{job_id}_all.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for stem_name, stem_info in final_stems.items():
            zf.write(stem_info['path'], f"{stem_name}.{output_format}")
    
    elapsed = time.time() - start_time
    total_size = sum(s['size_mb'] for s in final_stems.values())
    
    logger.info(f"[{job_id}] Completed in {elapsed:.1f}s - {len(final_stems)} stems, {total_size:.1f}MB total")
    
    return {
        "status": "completed",
        "input_size_mb": round(input_size / (1024 * 1024), 2),
        "stems": final_stems,
        "total_output_size_mb": round(total_size, 2),
        "zip_file": str(zip_path),
        "output_format": output_format,
        "engine": "MLX" if USE_MLX else "PyTorch",
        "processing_time_seconds": round(elapsed, 1)
    }


def download_youtube_audio(url: str, output_dir: Path, job_id: str) -> tuple:
    """Download audio from YouTube URL and return (audio_path, title)"""
    import time
    
    logger.info(f"[{job_id}] Downloading YouTube: {url}")
    start_time = time.time()
    
    output_path = output_dir / f"{job_id}"
    output_template = str(output_path)
    
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown download error"
            logger.error(f"[{job_id}] YouTube download failed: {error_msg}")
            raise Exception(f"YouTube download failed: {error_msg}")
        
        audio_file = output_dir / f"{job_id}.wav"
        
        if not audio_file.exists():
            downloaded_files = list(output_dir.glob(f"{job_id}.*"))
            if downloaded_files:
                audio_file = downloaded_files[0]
            else:
                raise Exception(f"Downloaded file not found. stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}")
        
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        title = f"YouTube_{job_id}"
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


@app.get("/")
async def health_check():
    """Health check endpoint"""
    global USE_MLX
    return {
        "status": "online",
        "service": "Stem Separation API v3",
        "model": MODEL_NAME,
        "optimization": "Apple Silicon (MLX)" if USE_MLX else "PyTorch",
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "youtube_support": True,
        "supported_output_formats": list(SUPPORTED_OUTPUT_FORMATS),
        "available_stems": STEMS,
        "default_output_format": "wav"
    }


@app.post("/youtube")
async def youtube_endpoint(
    url: str = Form(...),
    format: str = Form("wav"),
    stems: str = Form(None)
):
    """Download YouTube audio and extract stems
    
    Parameters:
    - url: YouTube video URL
    - format: Output format (wav, mp3, ogg, flac). Default: wav
    - stems: Comma-separated stems to extract (default: all). Options: vocals,drums,bass,other,guitar,piano
    
    Returns:
    - job_id: Use to check status and download results
    """
    
    output_format = format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )
    
    # Parse stems
    stems_list = None
    if stems:
        stems_list = [s.strip().lower() for s in stems.split(',')]
        invalid_stems = [s for s in stems_list if s not in STEMS]
        if invalid_stems:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stems: {invalid_stems}. Available: {', '.join(STEMS)}"
            )
    
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
    
    job_id = str(uuid.uuid4())[:8]
    
    jobs[job_id] = {
        "status": "downloading",
        "url": url,
        "output_format": output_format,
        "stems": stems_list if stems_list else "all",
        "started": asyncio.get_event_loop().time()
    }
    
    try:
        audio_path, title = await run_in_threadpool(
            download_youtube_audio,
            url,
            UPLOAD_DIR,
            job_id
        )
        
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["title"] = title
        
        result = await run_in_threadpool(
            separate_audio_all,
            audio_path,
            OUTPUT_DIR,
            job_id,
            output_format,
            stems_list
        )
        
        jobs[job_id].update(result)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "title": title,
            "output_format": output_format,
            "stems": stems_list if stems_list else "all"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/separate")
async def separate_endpoint(
    file: UploadFile = File(...),
    format: str = Form("wav"),
    stems: str = Form(None)
):
    """Upload audio file for stem extraction
    
    Parameters:
    - file: Audio file (mp3, wav, flac, ogg, m4a, aac, wma)
    - format: Output format (wav, mp3, ogg, flac). Default: wav
    - stems: Comma-separated stems to extract (default: all). Options: vocals,drums,bass,other,guitar,piano
    
    Returns:
    - job_id: Use to check status and download results
    """
    
    output_format = format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )
    
    stems_list = None
    if stems:
        stems_list = [s.strip().lower() for s in stems.split(',')]
        invalid_stems = [s for s in stems_list if s not in STEMS]
        if invalid_stems:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stems: {invalid_stems}. Available: {', '.join(STEMS)}"
            )
    
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_INPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_INPUT_FORMATS)}"
        )
    
    job_id = str(uuid.uuid4())[:8]
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
            )
        
        with open(input_path, 'wb') as f:
            f.write(content)
        
        jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "output_format": output_format,
            "stems": stems_list if stems_list else "all",
            "started": asyncio.get_event_loop().time()
        }
        
        result = await run_in_threadpool(
            separate_audio_all,
            str(input_path),
            OUTPUT_DIR,
            job_id,
            output_format,
            stems_list
        )
        
        jobs[job_id].update(result)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "output_format": output_format,
            "stems": stems_list if stems_list else "all"
        }
        
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
    
    if "stems" in job:
        response["stems"] = job["stems"]
    
    if "total_output_size_mb" in job:
        response["total_output_size_mb"] = job["total_output_size_mb"]
    
    if "output_format" in job:
        response["output_format"] = job["output_format"]
    
    if "processing_time_seconds" in job:
        response["processing_time_seconds"] = job["processing_time_seconds"]
    
    return response


@app.get("/download/{job_id}")
async def download_result(job_id: str, stem: str = None):
    """Download extracted stems
    
    Parameters:
    - job_id: Job ID
    - stem: Specific stem to download (vocals, drums, bass, other, guitar, piano). If not provided, returns zip of all stems.
    
    Returns:
    - Audio file (single stem) or ZIP file (all stems)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}"
        )
    
    output_format = job.get("output_format", "wav")
    media_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'ogg': 'audio/ogg',
        'flac': 'audio/flac'
    }
    
    filename_base = job.get("filename", "audio")
    if "title" in job:
        filename_base = job["title"]
    
    # If stem specified, return single stem
    if stem:
        stem = stem.lower()
        if stem not in STEMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stem. Available: {', '.join(STEMS)}"
            )
        
        stem_path = OUTPUT_DIR / f"{job_id}_{stem}.{output_format}"
        if not stem_path.exists():
            raise HTTPException(status_code=404, detail=f"Stem '{stem}' not found")
        
        return FileResponse(
            stem_path,
            media_type=media_types.get(output_format, "audio/wav"),
            filename=f"{Path(filename_base).stem}_{stem}.{output_format}"
        )
    
    # Return zip of all stems
    zip_path = OUTPUT_DIR / f"{job_id}_all.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip file not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{Path(filename_base).stem}_stems.zip"
    )


@app.get("/stems/{job_id}")
async def list_stems(job_id: str):
    """List available stems for a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}"
        )
    
    if "stems" not in job:
        raise HTTPException(status_code=400, detail="No stems found for this job")
    
    available = {}
    for stem_name, stem_info in job["stems"].items():
        available[stem_name] = {
            "size_mb": stem_info["size_mb"],
            "download_url": f"/download/{job_id}?stem={stem_name}"
        }
    
    return {
        "job_id": job_id,
        "stems": available,
        "zip_url": f"/download/{job_id}"
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Stem Separation API v3")
    parser.add_argument("--port", type=int, default=8766, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    logger.info(f"Starting Stem Separation API v3 on {args.host}:{args.port}")
    logger.info("Apple Silicon optimization: MLX enabled when available")
    logger.info(f"Available stems: {', '.join(STEMS)}")
    uvicorn.run(app, host=args.host, port=args.port)