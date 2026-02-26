import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class WaveOut:
    phi: torch.Tensor          # (B, T, d_model) — latent sequence
    dynamic_state: torch.Tensor # (B, T, d_out)   — learned wave/dynamic state

class MotionEncoder(nn.Module):
    """Transformer-based motion encoder for body trajectories or keypoints."""
    def __init__(self, d_in: int, d_model: int = 256, n_layers: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, d_model)"""
        return self.enc(self.proj(x))

class WavePose(nn.Module):
    """
    Simplified 2D-pose to Latent Wave Embeddings.
    Maps raw 2D-pose (or trajectories) to learned dynamic states.
    """
    def __init__(self, d_in: int, d_model: int = 256, d_out: int = 32,
                 n_layers: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.enc = MotionEncoder(d_in=d_in, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        
        # Head for learned dynamic state (the "wave analysis" part)
        # This can represent velocity, phase, etc. learned implicitly.
        self.head_wave = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_out)
        )

    def forward(self, tokens: torch.Tensor) -> WaveOut:
        """
        tokens: (B, T, D) e.g., (B, T, 34) for 17 x (x,y) COCO keypoints
        """
        phi = self.enc(tokens)                   # (B, T, d_model)
        dynamic_state = self.head_wave(phi)      # (B, T, d_out)
        return WaveOut(phi=phi, dynamic_state=dynamic_state)
