import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from MusclePose.models.wavepose import WaveOut

@dataclass
class PhysicsSummary:
    """Simplified wave-based motion summary."""
    exercise: str = ""
    quality_grade: str = ""
    smoothness: float = 0.0
    rep_count: int = 0
    wave_phases: List[Dict] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [
            f"Exercise: {self.exercise}",
            f"Quality: {self.quality_grade} (smoothness={self.smoothness:.2f})",
            f"Reps: {self.rep_count}",
        ]
        if self.wave_phases:
            phase_strs = [f"{w['type']}({w['duration_sec']:.2f}s)" for w in self.wave_phases[:6]]
            lines.append("Wave Phases: " + ", ".join(phase_strs))
        return "\n".join(lines)


def extract_physics_summary(clip_dict: Dict) -> PhysicsSummary:
    wf = clip_dict.get("wave_features", {})
    quality = wf.get("quality", {})
    harmonic = wf.get("harmonic", {})

    return PhysicsSummary(
        exercise=clip_dict.get("exercise", ""),
        quality_grade=quality.get("grade", ""),
        smoothness=quality.get("smoothness", 0.0),
        rep_count=harmonic.get("oscillation_count", 0),
        wave_phases=wf.get("waves", []),
    )


def build_llm_prompt(summary: PhysicsSummary, language_hint: str = "") -> str:
    system = (
        "You are an AI motion analyst specializing in wave-based latent analysis. "
        "Given the learned dynamic states from a 2D pose sequence, "
        "provide coaching feedback focusing on motion rhythm and technique."
    )
    user_block = summary.to_prompt()
    if language_hint:
        user_block += f"\n\nContext: {language_hint[:500]}"
    user_block += "\n\nProvide coaching feedback based on the learned wave states:"
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{user_block}<|end|>\n<|assistant|>\n"


class WaveBridge(nn.Module):
    """
    Learned Cross-Attention Bridge for WavePose.
    Uses learned event tokens to attend to the latent motion sequence.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_wave: int = 32,
        llm_dim: int = 3072,
        n_tokens: int = 8,
        n_heads: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        
        # Fuse phi + learned dynamic state
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_wave, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Sinusoidal PE for temporal context
        self.pe = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Learned event queries
        self.event_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        
        # Cross-Attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # LLM projection
        self.to_llm = nn.Linear(d_model, llm_dim)
        self.out_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, out: WaveOut) -> torch.Tensor:
        # Fuse latent phi with the dynamic wave states
        fused = self.fuse(torch.cat([out.phi, out.dynamic_state], dim=-1))
        
        # Add PE
        B, T, _ = fused.shape
        context = fused + self.pe[:, :T, :]
        
        # Event tokens attend to fused context
        queries = self.event_tokens.expand(B, -1, -1)
        soft_tokens, _ = self.attn(queries, context, context)
        soft_tokens = self.norm(queries + soft_tokens)
        
        # Project to LLM space
        raw = self.to_llm(soft_tokens)
        gate = torch.tanh(self.out_gate)
        return raw * gate * 0.03 # scale to typical LLM embedding variance
