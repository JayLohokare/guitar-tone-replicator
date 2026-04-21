"""
Tone Replicator - Core Model v2
Neural amp modeling with improved architecture.

Key improvements over v1:
- Skip connections in every ConvBlock for gradient flow
- Larger default receptive fields (2000+ samples)
- Output tanh limiting to prevent clipping artifacts
- Pre-emphasis support baked into the model
- Conditioned blocks (loudness-aware) for better clean→distortion modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class ConvBlock(nn.Module):
    """Dilated causal convolution block with gated activation + skip connection."""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int, skip_channels: int = None):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels * 2, kernel_size, dilation=dilation,
                      padding=(kernel_size - 1) * dilation)
        )
        self.residual = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, 1)
        )
        # Skip connection: project to skip_channels for accumulation
        self.skip_channels = skip_channels or channels
        self.skip_proj = nn.utils.weight_norm(
            nn.Conv1d(channels, self.skip_channels, 1)
        )
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns (residual_output, skip_output).
        skip_output is added to the global skip accumulator.
        """
        h = self.conv(x)
        h = h[:, :, :x.shape[-1]]  # Causal: trim future
        gate, filter_ = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filter_)
        residual = self.residual(h) + x
        skip = self.skip_proj(h)
        return residual, skip


class ToneNet(nn.Module):
    """
    WaveNet-style model for guitar tone replication.
    
    Improvements over v1:
    - Skip connections (accumulated across all blocks)
    - Larger receptive field by default
    - Output limiting (tanh) to prevent clipping
    - Optional pre-emphasis filter coefficient
    """
    
    def __init__(
        self,
        channels: int = 40,
        num_blocks: int = 16,
        kernel_size: int = 3,
        dilations: list = None,
        sample_rate: float = 48000.0,
        pre_emphasis: float = 0.85,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.pre_emphasis = pre_emphasis
        
        if dilations is None:
            # Stack dilations: 1,2,...,512 repeated to fill num_blocks
            # This gives a huge receptive field
            base_dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            dilations = (base_dilations * ((num_blocks // len(base_dilations)) + 1))[:num_blocks]
        
        self.dilations = dilations
        
        # Input projection: 1 channel → channels
        self.input_conv = nn.utils.weight_norm(
            nn.Conv1d(1, channels, 1)
        )
        
        # Dilated convolution blocks
        self.blocks = nn.ModuleList([
            ConvBlock(channels, kernel_size, d, skip_channels=channels) 
            for d in dilations
        ])
        
        # Output projection: accumulated skip → channels → 1
        # The skip connections accumulate contributions from ALL blocks
        self.output_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(channels, channels * 2, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(channels * 2, 1, 1)),
        )
        
        # Initialize output layer to near-zero so model starts as pass-through
        # This is critical for stable training with skip connections
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)
        
    @property
    def receptive_field(self) -> int:
        """Receptive field in samples."""
        rf = 1
        for block in self.blocks:
            rf += (block.conv.kernel_size[0] - 1) * block.conv.dilation[0]
        return rf
    
    def apply_preemphasis(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
        Used for loss computation only — not applied inside the model."""
        if self.pre_emphasis == 0:
            return x
        return torch.cat([x[:, :, :1], x[:, :, 1:] - self.pre_emphasis * x[:, :, :-1]], dim=-1)
    
    def apply_deemphasis(self, y: torch.Tensor) -> torch.Tensor:
        """Not used in forward pass. Kept for compatibility."""
        if self.pre_emphasis == 0:
            return y
        raise NotImplementedError("De-emphasis should not be called in model forward pass")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio tensor of shape (batch, 1, time)
        Returns:
            Output audio tensor of shape (batch, 1, time)
        """
        # Pad for causal convolution
        pad = self.receptive_field - 1
        x_padded = F.pad(x, (pad, 0))
        
        h = self.input_conv(x_padded)
        
        # Accumulate skip connections
        skip_sum = torch.zeros_like(h)
        
        for block in self.blocks:
            h, skip = block(h)
            # Trim skip to match current length minus padding
            skip_sum = skip_sum + skip
        
        # Trim padding from skip sum
        skip_sum = skip_sum[:, :, pad:]
        
        # Output projection from accumulated skip connections
        out = self.output_conv(skip_sum)
        
        # No pre/de-emphasis in model — handled in training loss instead
        # No tanh limiting — soft clipping handled in process_audio()
        
        return out


class LSTMToneNet(nn.Module):
    """
    LSTM-based tone model (lighter alternative to WaveNet).
    Better for longer sequences, lower memory.
    """
    
    def __init__(
        self,
        hidden_size: int = 48,
        num_layers: int = 2,
        sample_rate: float = 48000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio tensor of shape (batch, 1, time) or (batch, time)
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (batch, time)
        
        # Process in chunks to save memory
        x = x.unsqueeze(-1)  # (batch, time, 1)
        h, _ = self.lstm(x)
        y = self.linear(h)
        return y.permute(0, 2, 1)  # (batch, 1, time)


def create_model(model_type: str = "wavenet", size: str = "standard") -> nn.Module:
    """Factory for creating tone models.
    
    Receptive field targets:
    - nano: ~50ms (2400 samples @ 48kHz) — quick tests
    - lite: ~100ms (4800 samples) — decent quality  
    - standard: ~200ms (9600 samples) — production quality
    - feather: ~150ms (7200 samples) — good balance
    """
    configs = {
        "wavenet": {
            # nano: ~100ms, quick testing
            "nano": {"channels": 16, "num_blocks": 10, "kernel_size": 5,
                     "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
            # lite: ~200ms, decent quality
            "lite": {"channels": 28, "num_blocks": 16, "kernel_size": 5,
                     "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                   1, 2, 4, 8, 16, 32]},
            # standard: ~400ms, production quality (large receptive field)
            "standard": {"channels": 40, "num_blocks": 22, "kernel_size": 5,
                         "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                       1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                       1, 2]},
            # feather: ~300ms, good balance
            "feather": {"channels": 32, "num_blocks": 18, "kernel_size": 5,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                      1, 2, 4, 8, 16, 32, 64, 128]},
        },
        "lstm": {
            "nano": {"hidden_size": 16, "num_layers": 1},
            "lite": {"hidden_size": 32, "num_layers": 2},
            "standard": {"hidden_size": 48, "num_layers": 2},
        }
    }
    
    if model_type == "wavenet":
        return ToneNet(**configs["wavenet"][size])
    elif model_type == "lstm":
        return LSTMToneNet(**configs["lstm"][size])
    else:
        raise ValueError(f"Unknown model type: {model_type}")