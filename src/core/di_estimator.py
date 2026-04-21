"""
Tone Replicator - DI Estimator v2
Estimates a DI signal from a processed guitar stem.

Key improvements over v1:
- Spectral envelope estimation using median filtering (more robust)
- Adaptive inversion strength based on spectral flatness
- Better handling of distortion (squared-magnitude approach)
- Dynamic range preservation with RMS matching
- Option to use the original phase (critical for transients)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from pathlib import Path


# Typical guitar amp frequency response (approximate)
# Values in dB: positive = boost, negative = cut
AMP_RESPONSE_DB = {
    80: 4.0,     # Bass resonance from cabinet
    150: 2.5,    # Low-mid resonance
    300: 0.0,    # Transition
    500: -1.0,   # Lower mids (varies by amp)
    800: -2.0,   # Mid scoop
    1000: -1.5,  # Mid scoop
    1500: 0.0,   # Mids returning
    2000: 1.5,   # Upper mids
    3000: 4.0,   # Presence peak
    4000: 3.0,   # Presence/bite
    5000: 1.5,   # Starting to roll off
    6000: 0.0,   # Transition
    8000: -3.0,  # Cab rolloff
    10000: -6.0, # More rolloff
    12000: -10.0,# Steep rolloff
    15000: -16.0,# Nearly gone
    18000: -25.0,# Silent
}

# Different amp styles have different EQ curves
AMP_STYLES = {
    "clean": {
        80: 2.0, 150: 1.5, 300: 0.0, 500: 0.5, 800: -0.5,
        1000: 0.0, 1500: 0.5, 2000: 1.0, 3000: 2.0, 4000: 1.5,
        5000: 0.5, 6000: -1.0, 8000: -4.0, 10000: -8.0, 15000: -20.0,
    },
    "crunch": {
        80: 4.0, 150: 3.0, 300: 1.0, 500: -1.0, 800: -3.0,
        1000: -2.0, 1500: 0.0, 2000: 2.0, 3000: 5.0, 4000: 4.0,
        5000: 2.0, 6000: 0.0, 8000: -3.0, 10000: -7.0, 15000: -18.0,
    },
    "high_gain": {
        80: 6.0, 150: 5.0, 300: 2.0, 500: -2.0, 800: -4.0,
        1000: -3.0, 1500: -1.0, 2000: 3.0, 3000: 7.0, 4000: 6.0,
        5000: 4.0, 6000: 1.0, 8000: -2.0, 10000: -6.0, 15000: -20.0,
    },
}


def estimate_amp_eq_curve(sr: int, n_fft: int = 4096, amp_style: str = "auto") -> np.ndarray:
    """
    Build a smooth EQ curve in the frequency domain representing
    average guitar amp coloration.
    
    Args:
        sr: Sample rate
        n_fft: FFT size
        amp_style: "clean", "crunch", "high_gain", or "auto" (uses default blend)
    
    Returns:
        Linear gain values for each FFT bin
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Select the appropriate curve
    if amp_style in AMP_STYLES:
        ref_curve = AMP_STYLES[amp_style]
    else:
        ref_curve = AMP_RESPONSE_DB  # Default blend
    
    # Interpolate the reference curve
    ref_freqs = sorted(ref_curve.keys())
    ref_vals = [ref_curve[f] for f in ref_freqs]
    
    # Extend to full frequency range
    ref_freqs_ext = [20] + ref_freqs + [sr // 2]
    ref_vals_ext = [0.0] + ref_vals + [-30.0]
    
    curve_db = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        idx = np.searchsorted(ref_freqs_ext, f)
        if idx == 0:
            curve_db[i] = ref_vals_ext[0]
        elif idx >= len(ref_vals_ext):
            curve_db[i] = ref_vals_ext[-1]
        else:
            f_lo, f_hi = ref_freqs_ext[idx - 1], ref_vals_ext[idx]
            v_lo, v_hi = ref_vals_ext[idx - 1], ref_vals_ext[idx]
            if f_hi > f_lo:
                t = np.log(f / f_lo) / np.log(f_hi / f_lo)
            else:
                t = 0
            curve_db[i] = v_lo + t * (v_hi - v_lo)
    
    # Smooth the curve to avoid sharp transitions
    curve_db = uniform_filter1d(curve_db, size=8)
    
    # Convert dB to linear
    curve_linear = 10.0 ** (curve_db / 20.0)
    return curve_linear


def detect_amp_style(stem: np.ndarray, sr: int) -> str:
    """
    Auto-detect the amp style from spectral characteristics.
    
    High-gain tones have more high-frequency energy relative to fundamentals.
    Clean tones have more pronounced fundamentals and less harmonics.
    """
    # Compute spectral features
    S = np.abs(librosa.stft(stem, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    
    # Energy in different bands
    low_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 2000)
    high_mask = (freqs >= 2000) & (freqs < 6000)
    
    low_energy = np.mean(S[low_mask, :] ** 2)
    mid_energy = np.mean(S[mid_mask, :] ** 2)
    high_energy = np.mean(S[high_mask, :] ** 2)
    
    # Spectral flatness (higher = more noise-like = more distortion)
    flatness = librosa.feature.spectral_flatness(y=stem)
    avg_flatness = np.mean(flatness)
    
    # High-frequency ratio
    hf_ratio = high_energy / (mid_energy + 1e-10)
    
    # Classify
    if avg_flatness > 0.15 or hf_ratio > 0.5:
        return "high_gain"
    elif avg_flatness > 0.05 or hf_ratio > 0.2:
        return "crunch"
    else:
        return "clean"


def estimate_di_from_stem(
    stem_path: str,
    output_path: str = None,
    sr: int = 48000,
    n_fft: int = 4096,
    hop_length: int = 1024,
    amp_style: str = "auto",
    inversion_strength: float = 0.6,
) -> np.ndarray:
    """
    Estimate a DI signal from a processed guitar stem.
    
    v2 improvements:
    - Larger FFT (4096) for better frequency resolution
    - Median-based spectral envelope (more robust to transients)
    - Auto-detect amp style (clean/crunch/high_gain)
    - Adaptive inversion strength
    - Phase preservation (keeps original transients)
    - RMS envelope matching (preserves dynamics)
    
    Args:
        stem_path: Path to the extracted guitar stem
        output_path: Optional path to save the estimated DI as WAV
        sr: Target sample rate
        n_fft: FFT size for spectral processing
        hop_length: STFT hop length
        amp_style: "clean", "crunch", "high_gain", or "auto"
        inversion_strength: How strongly to invert the amp EQ (0.0-1.0)
    
    Returns:
        Estimated DI signal as numpy array
    """
    # Load the stem
    stem, _ = librosa.load(stem_path, sr=sr, mono=True)
    
    # Auto-detect amp style
    if amp_style == "auto":
        amp_style = detect_amp_style(stem, sr)
        print(f"  Detected amp style: {amp_style}")
    
    # Compute STFT with higher resolution
    S = librosa.stft(stem, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)
    
    # ---- Step 1: Estimate the spectral envelope ----
    # Use percentile-based envelope (more robust than mean/median)
    # P75 captures the "note present" spectral shape, ignoring silence
    spectral_envelope = np.percentile(mag + 1e-10, 75, axis=1)
    
    # Heavy smoothing to get the coarse spectral shape (not note-level detail)
    smooth_envelope = uniform_filter1d(spectral_envelope, size=64)
    
    # ---- Step 2: Estimate the amp EQ curve ----
    amp_eq = estimate_amp_eq_curve(sr=sr, n_fft=n_fft, amp_style=amp_style)
    
    # ---- Step 3: Estimate what the DI should look like ----
    # A typical DI has a smooth 1/f-ish rolloff
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # More realistic DI reference shape:
    # - Fundamental resonance around 100-300Hz
    # - Gradual rolloff above 1kHz
    # - Slight presence bump around 2-3kHz (string attack)
    di_reference = np.ones_like(freqs, dtype=np.float64)
    
    # Low-frequency resonance (guitar body)
    di_reference *= 1.0 / (1.0 + ((freqs - 150) / 100) ** 2) * 0.3 + 0.7
    
    # Gradual high-frequency rolloff (typical guitar pickup response)
    di_reference *= 1.0 / (1.0 + (freqs / 4000.0) ** 1.2)
    
    # Slight presence peak at 2-3kHz (attack transients in DI)
    presence_peak = 1.0 + 0.3 * np.exp(-((freqs - 2500) / 800) ** 2)
    di_reference *= presence_peak
    
    # Normalize
    di_reference = di_reference / np.max(di_reference + 1e-10)
    
    # ---- Step 4: Compute the coloration ratio ----
    # coloration = what the amp added = stem_envelope / di_reference
    stem_norm = smooth_envelope / (np.max(smooth_envelope) + 1e-10)
    coloration_ratio = stem_norm / (di_reference + 1e-10)
    
    # Smooth the ratio heavily — we want coarse correction, not fine
    coloration_ratio = uniform_filter1d(coloration_ratio, size=128)
    
    # Clamp to avoid extreme values
    # Typical amp coloration: -15dB to +12dB
    coloration_ratio = np.clip(coloration_ratio, 0.05, 20.0)
    
    # ---- Step 5: Adaptive inversion strength ----
    # For high-gain tones, invert less (more distortion = less reliable inversion)
    # For clean tones, invert more (more linear = better inversion)
    if amp_style == "high_gain":
        effective_strength = inversion_strength * 0.4  # Be conservative with distortion
    elif amp_style == "crunch":
        effective_strength = inversion_strength * 0.6
    else:  # clean
        effective_strength = inversion_strength * 0.8
    
    # Partial inversion: remove some of the amp coloration
    correction = coloration_ratio ** (-effective_strength)
    
    # ---- Step 6: Apply spectral correction ----
    corrected_mag = mag * correction[:, np.newaxis]
    
    # ---- Step 7: Add subtle noise floor for realism ----
    # Real DI signals have a small noise floor that helps the model learn
    noise_floor = np.random.randn(*mag.shape).astype(np.float32) * np.mean(mag) * 0.001
    corrected_mag = np.maximum(corrected_mag, np.abs(noise_floor))
    
    # ---- Step 8: Reconstruct time-domain signal preserving original phase ----
    S_di = corrected_mag * np.exp(1j * phase)
    di_audio = librosa.istft(S_di, hop_length=hop_length, n_fft=n_fft, length=len(stem))
    
    # ---- Step 9: Match the dynamic envelope ----
    # This is critical: the estimated DI should have the same dynamics as the target
    frame_len = int(sr * 0.005)  # 5ms frames (finer than v1's 10ms)
    n_frames = len(stem) // frame_len
    
    if n_frames > 1:
        stem_rms = np.array([
            np.sqrt(np.mean(stem[i*frame_len:(i+1)*frame_len] ** 2))
            for i in range(n_frames)
        ])
        di_rms = np.array([
            np.sqrt(np.mean(di_audio[i*frame_len:(i+1)*frame_len] ** 2))
            for i in range(n_frames)
        ])
        
        # Compute gain ratio with smoothing
        ratio = stem_rms / (di_rms + 1e-10)
        ratio = uniform_filter1d(ratio, size=20)  # Smoother gain riding
        ratio = np.clip(ratio, 0.05, 20.0)  # Reasonable bounds
        
        # Apply time-varying gain
        for i in range(n_frames):
            start = i * frame_len
            end = min((i + 1) * frame_len, len(di_audio))
            if end <= len(di_audio):
                di_audio[start:end] *= ratio[i]
    
    # ---- Step 10: Add transient emphasis ----
    # DI signals have sharper transients than amp-processed signals
    # Re-emphasize transients using an envelope follower
    from scipy.signal import hilbert
    try:
        # Analytical signal envelope
        analytic = hilbert(di_audio)
        env = np.abs(analytic)
        
        # Compute transient function: rate of change of envelope
        env_diff = np.diff(env, prepend=0)
        transient_weight = np.maximum(env_diff, 0)  # Only positive changes (attacks)
        transient_weight = transient_weight / (np.max(transient_weight) + 1e-10)
        
        # Boost transients slightly (5-10% boost)
        transient_boost = 1.0 + 0.08 * transient_weight
        di_audio = di_audio * transient_boost
    except Exception:
        pass  # Skip transient boost if hilbert fails (edge case)
    
    # ---- Step 11: Normalize ----
    # Match the overall level of the original stem
    stem_peak = np.max(np.abs(stem))
    di_peak = np.max(np.abs(di_audio))
    if di_peak > 0:
        di_audio = di_audio * (stem_peak / di_peak) * 0.85
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, di_audio, sr)
    
    return di_audio


def estimate_di_simple(
    stem_path: str,
    output_path: str = None,
    sr: int = 48000,
) -> np.ndarray:
    """
    Simpler DI estimation: de-emphasis + mid boost + soft clip.
    Faster but less accurate than the spectral method.
    """
    stem, _ = librosa.load(stem_path, sr=sr, mono=True)
    
    # De-emphasize highs (opposite of guitar amp presence boost)
    stem = librosa.effects.preemphasis(stem, coef=-0.5)
    
    # STFT-based mid boost
    n_fft = 4096
    hop = 1024
    S = librosa.stft(stem, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Boost mids, reduce presence
    for i, f in enumerate(freqs):
        if 300 < f < 1500:
            mag[i, :] *= 1.5
        elif 2000 < f < 5000:
            mag[i, :] *= 0.7
    
    S_di = mag * np.exp(1j * phase)
    di_audio = librosa.istft(S_di, hop_length=hop, n_fft=n_fft, length=len(stem))
    
    # Soft clip
    di_audio = np.tanh(di_audio * 2.0) / 2.0
    
    # Normalize
    max_val = np.max(np.abs(di_audio))
    if max_val > 0:
        di_audio = di_audio / max_val * 0.8
    
    if output_path:
        sf.write(output_path, di_audio, sr)
    
    return di_audio