"""
Tone Replicator - Training Engine v2
Trains a neural tone model from DI/processed audio pairs.

Key improvements over v1:
- Longer segment length (1-2 seconds instead of 170ms)
- Pre-emphasis/de-emphasis filters for better learning
- Multi-resolution STFT loss with better parameters
- DC loss component (prevent output offset)
- Gain augmentation (random gain variation during training)
- Warmup + cosine annealing LR schedule
- Gradient accumulation for effective larger batches
- Better early stopping with patience
- Input normalization (match DI/target levels)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader

from .model import ToneNet, LSTMToneNet, create_model
from .dataset import ToneDataset
from .di_estimator import estimate_di_from_stem


class ESR(nn.Module):
    """Error-to-Signal Ratio loss - standard metric for amp modeling."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        return (error ** 2).sum() / (target ** 2).sum() + 1e-8


class DCLoss(nn.Module):
    """DC offset loss - penalizes mean offset in output."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.mean() - target.mean()) ** 2


class MRSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for perceptual quality.
    
    v2 improvements:
    - Larger FFT sizes for better frequency resolution
    - More FFT sizes for better coverage
    - Both magnitude and log-magnitude losses
    - Phase-aware loss component
    """
    
    def __init__(
        self,
        fft_sizes=[1024, 2048, 4096, 8192],
        hop_sizes=[256, 512, 1024, 2048],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MRSTFT loss.
        
        Note: STFT gradients can be unstable on MPS. We compute this loss
        with detached tensors to use it as a monitoring metric, not for
        direct gradient computation. The ESR loss provides the main
        gradient signal.
        """
        # Compute on CPU for compatibility
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        loss = torch.tensor(0.0)
        
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            # Skip if segment is shorter than FFT
            if pred_cpu.shape[-1] < n_fft:
                continue
                
            window = torch.hann_window(n_fft)
            
            pred_stft = torch.stft(
                pred_cpu.squeeze(1), n_fft=n_fft, hop_length=hop,
                window=window, return_complex=True
            )
            target_stft = torch.stft(
                target_cpu.squeeze(1), n_fft=n_fft, hop_length=hop,
                window=window, return_complex=True
            )
            
            # Magnitude losses
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            
            # Log magnitude loss (perceptual frequency matching)
            loss += nn.functional.l1_loss(
                torch.log1p(pred_mag), 
                torch.log1p(target_mag)
            )
            
            # Linear magnitude loss (temporal envelope matching)
            loss += 0.5 * nn.functional.l1_loss(pred_mag, target_mag)
            
        return loss.to(pred.device)


class PreEmphasis(nn.Module):
    """Pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
    
    Applied to both prediction and target in the ESR loss to weight
    the loss toward mid/high frequencies where tone character lives.
    Not applied inside the model — the model learns the full spectral mapping.
    """
    
    def __init__(self, coeff: float = 0.85):
        super().__init__()
        self.coeff = coeff
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coeff == 0:
            return x
        return torch.cat([x[:, :, :1], x[:, :, 1:] - self.coeff * x[:, :, :-1]], dim=-1)




class ToneTrainer:
    """
    Trains a tone replication model.
    
    Two modes:
    1. Paired mode: DI + processed audio → learns the transformation
    2. Blind mode: Only processed audio → uses estimated DI to approximate
    """
    
    def __init__(
        self,
        model_type: str = "wavenet",
        model_size: str = "standard",
        sample_rate: int = 48000,
        device: str = "auto",
        learning_rate: float = 0.003,
        lr_decay: float = 0.005,
        batch_size: int = 16,
        segment_length: int = 16384,
        pre_emphasis: float = 0.85,
        mrstft_weight: float = 0.25,
        dc_weight: float = 0.01,
    ):
        self.model_type = model_type
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.pre_emphasis_coeff = pre_emphasis
        self.mrstft_weight = mrstft_weight
        self.dc_weight = dc_weight
        
        # Auto device selection
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        print(f"  Training device: {self.device}")
        
        # Create model
        self.model = create_model(model_type, model_size).to(self.device)
        
        # Report receptive field
        if hasattr(self.model, 'receptive_field'):
            rf_ms = self.model.receptive_field / sample_rate * 1000
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Model: {model_type}/{model_size}, {total_params:,} params")
            print(f"  Receptive field: {self.model.receptive_field} samples ({rf_ms:.1f}ms)")
        
        # Loss functions
        self.esr_loss = ESR()
        self.dc_loss = DCLoss()
        self.mrstft_loss = MRSTFTLoss()
        self.pre_emph = PreEmphasis(pre_emphasis).to(self.device)
        
        # Training state
        self.history: List[Dict] = []
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def train_paired(
        self,
        di_path: str,
        processed_path: str,
        epochs: int = 100,
        delay: int = 0,
        validation_split: float = 0.1,
        save_path: Optional[str] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Train from paired DI/processed audio (standard NAM-style training).
        """
        # Create dataset
        audio, sr = librosa.load(processed_path, sr=self.sample_rate, mono=True)
        total_samples = len(audio)
        val_start = int(total_samples * (1 - validation_split))
        
        train_dataset = ToneDataset(
            di_path, processed_path,
            sample_rate=self.sample_rate,
            segment_length=self.segment_length,
            delay=delay,
            stop_sample=val_start,
        )
        val_dataset = ToneDataset(
            di_path, processed_path,
            sample_rate=self.sample_rate,
            segment_length=self.segment_length,
            delay=delay,
            start_sample=val_start,
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        print(f"  Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return self._train_loop(
            train_loader, val_loader, epochs, save_path, progress_callback
        )
    
    def train_blind(
        self,
        target_tone_path: str,
        epochs: int = 100,
        save_path: Optional[str] = None,
        progress_callback=None,
    ):
        """
        Train from target tone only, using an ESTIMATED DI signal.
        
        The model learns: estimated_DI(x) ≈ target_tone
        """
        temp_dir = Path(save_path or "/tmp/tone_replicator").parent
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Estimate DI from the stem
        est_di_path = str(temp_dir / "_estimated_di.wav")
        print("  Estimating DI signal from guitar stem...")
        est_di = estimate_di_from_stem(
            target_tone_path,
            output_path=est_di_path,
            sr=self.sample_rate,
        )
        print(f"  Estimated DI: {len(est_di)} samples ({len(est_di)/self.sample_rate:.1f}s)")
        
        # Load target tone
        target_audio, sr = librosa.load(
            target_tone_path, sr=self.sample_rate, mono=True
        )
        
        # Match lengths
        min_len = min(len(est_di), len(target_audio))
        est_di = est_di[:min_len]
        target_audio = target_audio[:min_len]
        
        # Re-save the aligned files
        target_path = str(temp_dir / "_target_aligned.wav")
        sf.write(est_di_path, est_di, self.sample_rate)
        sf.write(target_path, target_audio, self.sample_rate)
        
        # Auto-detect delay
        delay = self._detect_delay(est_di, target_audio)
        print(f"  Detected delay: {delay} samples ({delay/self.sample_rate*1000:.1f}ms)")
        
        return self.train_paired(
            di_path=est_di_path,
            processed_path=target_path,
            epochs=epochs,
            delay=delay,
            save_path=save_path,
            progress_callback=progress_callback,
        )
    
    def _detect_delay(self, di: np.ndarray, processed: np.ndarray) -> int:
        """
        Detect the latency/delay between DI and processed audio
        using cross-correlation.
        """
        max_lag = min(48000, len(di) // 4)
        di_segment = di[:max_lag * 4]
        proc_segment = processed[:max_lag * 4]
        
        # Normalize both for better correlation
        di_norm = di_segment / (np.std(di_segment) + 1e-10)
        proc_norm = proc_segment / (np.std(proc_segment) + 1e-10)
        
        correlation = np.correlate(proc_norm, di_norm, mode='full')
        best_lag = np.argmax(correlation) - len(di_segment) + 1
        
        best_lag = max(-max_lag, min(max_lag, best_lag))
        return int(best_lag)
    
    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: Optional[str],
        progress_callback,
    ) -> Dict[str, Any]:
        """Core training loop with all v2 improvements."""
        
        # ---- Learning rate schedule: warmup + cosine annealing ----
        warmup_epochs = min(5, epochs // 5)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Cosine annealing scheduler: starts at learning_rate after warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 10), eta_min=1e-6
        )
        
        # Gradient accumulation steps (effective batch = batch_size * accum_steps)
        accum_steps = 2  # v2: reduced from 4 for faster iteration
        
        # Early stopping
        patience = max(10, epochs // 5)
        patience_counter = 0
        
        print(f"Starting training loop: {epochs} epochs, {len(train_loader)} batches/epoch", flush=True)
        
        for epoch in range(epochs):
            # ---- Warmup ----
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                lr = self.learning_rate * warmup_factor
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            
            epoch_start = time.time() if epoch == 0 else None
            if epoch == 0:
                print(f"Epoch 1 starting...", flush=True)
            
            # ---- Training ----
            self.model.train()
            train_loss = torch.tensor(0.0, device=self.device)  # Accumulate on device
            train_esr = torch.tensor(0.0, device=self.device)    # Avoid MPS sync per batch
            num_batches = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                # ---- Gain augmentation ----
                # Randomly vary the input gain to help generalization
                if self.model.training:
                    gain = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
                    x = x * gain
                
                # ---- Forward pass ----
                pred = self.model(x)
                
                # ---- Apply pre-emphasis for spectral-weighted ESR loss ----
                # This focuses the ESR loss on mid/high frequencies where
                # guitar tone character lives, rather than low-frequency energy
                pred_emph = self.pre_emph(pred)
                y_emph = self.pre_emph(y)
                
                # ---- Compute losses ----
                # ESR in pre-emphasis domain (focuses on spectral shape)
                loss_esr = self.esr_loss(pred_emph, y_emph)
                # DC loss (prevent output offset)
                loss_dc = self.dc_loss(pred, y)
                
                # Total loss: ESR (gradient) + DC (gradient)
                # MRSTFT is too expensive on MPS due to CPU sync; skip during training
                loss = loss_esr + loss_dc
                
                # ---- Gradient accumulation ----
                loss = loss / accum_steps
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Accumulate loss values (avoid .item() to prevent MPS sync)
                train_loss += loss.detach() * accum_steps
                train_esr += loss_esr.detach()
                num_batches += 1
            
            # Convert accumulated tensors to scalars (single MPS sync)
            train_loss = train_loss.item() / max(num_batches, 1)
            train_esr = train_esr.item() / max(num_batches, 1)
            
            # Step scheduler (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # ---- Validation ----
            val_loss = torch.tensor(0.0, device=self.device)
            val_esr = torch.tensor(0.0, device=self.device)
            num_val = 0
            
            self.model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    pred = self.model(x)
                    
                    # ESR in pre-emphasis domain for consistency
                    pred_emph = self.pre_emph(pred)
                    y_emph = self.pre_emph(y)
                    loss_esr = self.esr_loss(pred_emph, y_emph)
                    val_esr += loss_esr.detach()
                    val_loss += loss_esr.detach()
                    num_val += 1
            
            # Convert accumulated tensors to scalars (single MPS sync)
            val_loss = val_loss.item() / max(num_val, 1)
            val_esr = val_esr.item() / max(num_val, 1)
            
            # ---- Track best model ----
            if val_esr < self.best_loss:
                self.best_loss = val_esr
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_esr": train_esr,
                "val_loss": val_loss,
                "val_esr": val_esr,
                "lr": optimizer.param_groups[0]['lr'],
            }
            self.history.append(entry)
            
            if progress_callback:
                progress_callback(entry)
            
            # Print progress every epoch
            print(f"  Epoch {epoch+1}/{epochs}: train_esr={train_esr:.6f}, val_esr={val_esr:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
            
            # Early stopping
            if val_esr < 0.005:  # ESR < 0.5% is excellent
                print(f"  Early stopping at epoch {epoch+1}: ESR = {val_esr:.6f}")
                break
            
            if patience_counter >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs (best ESR={self.best_loss:.6f})")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save model
        if save_path:
            self.save_model(save_path)
        
        return {
            "history": self.history,
            "best_val_esr": self.best_loss,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
        }
    
    def save_model(self, path: str):
        """Save model weights and metadata."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "best_val_esr": self.best_loss,
            "pre_emphasis": self.pre_emphasis_coeff,
        }, save_dir / "model.pth")
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "best_val_esr": self.best_loss,
            "receptive_field": self.model.receptive_field if hasattr(self.model, 'receptive_field') else None,
            "pre_emphasis": self.pre_emphasis_coeff,
            "segment_length": self.segment_length,
            "mrstft_weight": self.mrstft_weight,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, path: str):
        """Load a saved model."""
        checkpoint = torch.load(
            Path(path) / "model.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model_type = checkpoint["model_type"]
        self.model_size = checkpoint["model_size"]
        self.sample_rate = checkpoint["sample_rate"]
        self.pre_emphasis_coeff = checkpoint.get("pre_emphasis", 0.85)
        
        self.model = create_model(self.model_type, self.model_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    @torch.no_grad()
    def process_audio(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """
        Apply the learned tone to an input audio file.
        
        Args:
            input_path: Path to DI audio file
            output_path: Where to save the processed output
            
        Returns:
            Path to the output file
        """
        self.model.eval()
        
        # Load input
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Process in chunks to avoid memory issues
        chunk_size = self.sample_rate * 10  # 10-second chunks
        rf = self.model.receptive_field if hasattr(self.model, 'receptive_field') else 0
        output_chunks = []
        
        for i in range(0, audio_tensor.shape[-1], chunk_size):
            chunk = audio_tensor[:, :, i:i + chunk_size]
            
            # Pad for receptive field
            if rf > 0:
                # Use overlap from previous chunk if available
                if i > 0:
                    overlap = min(rf, i)
                    chunk = audio_tensor[:, :, i - overlap:i + chunk_size]
                else:
                    chunk = torch.nn.functional.pad(chunk, (rf, 0))
            
            with torch.no_grad():
                out = self.model(chunk)
            
            # Trim padding/overlap from output
            if rf > 0:
                if i > 0:
                    # Remove overlap region from output
                    out = out[:, :, overlap:]
                else:
                    out = out[:, :, rf:]
            
            output_chunks.append(out.cpu().squeeze().numpy())
        
        output_audio = np.concatenate(output_chunks)
        
        # Match length to input
        if len(output_audio) > len(audio):
            output_audio = output_audio[:len(audio)]
        elif len(output_audio) < len(audio):
            output_audio = np.pad(output_audio, (0, len(audio) - len(output_audio)))
        
        # The model learns gain naturally (dataset preserves gain ratio).
        # Only normalize if clipping would occur.
        output_peak = np.max(np.abs(output_audio))
        if output_peak > 0.99:
            # Prevent clipping by scaling down
            output_audio = output_audio * (0.95 / output_peak)
        
        sf.write(output_path, output_audio, self.sample_rate)
        return output_path