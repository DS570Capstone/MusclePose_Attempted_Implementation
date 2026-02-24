"""
Bridge between MusclePose physics pipeline and Wave-LLM.

Flow:
  clip JSON  ->  ClipDataset  ->  trajectories (C, T)
                                     |
                           PhysicsSummary (joint angles, torques, etc.)
                                     |
                           WaveEncoder  ->  wave embeddings
                                     |
                           WaveToTextBridge  ->  LLM soft tokens
                                     |
                           LLM generates coaching text
"""

import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhysicsSummary:
    exercise: str = ""
    fps: float = 30.0
    n_frames: int = 0
    expert: bool = False
    quality_grade: str = ""
    smoothness: float = 0.0
    control: float = 0.0
    efficiency_pct: float = 0.0
    consistency: float = 0.0
    damping_ratio: float = 0.0
    dominant_hz: float = 0.0
    rep_count: int = 0
    peak_power_w: float = 0.0
    wave_phases: List[Dict] = field(default_factory=list)
    error_rate: list = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [
            f"Exercise: {self.exercise}",
            f"Expert: {self.expert}",
            f"Reps: {self.rep_count}",
            f"Quality: {self.quality_grade} (smoothness={self.smoothness:.2f}, control={self.control:.2f}, "
            f"efficiency={self.efficiency_pct:.1f}%, consistency={self.consistency:.2f})",
            f"Peak power: {self.peak_power_w:.1f} W",
            f"Damping ratio: {self.damping_ratio:.4f}",
            f"Dominant frequency: {self.dominant_hz:.3f} Hz",
        ]
        if self.wave_phases:
            phase_strs = []
            for w in self.wave_phases[:6]:
                phase_strs.append(f"{w['type']}({w['duration_sec']:.2f}s, v={w['mean_velocity']:.2f})")
            lines.append("Phases: " + ", ".join(phase_strs))
        if self.error_rate:
            lines.append(f"Errors: {self.error_rate}")
        return "\n".join(lines)


def extract_physics_summary(clip_dict: Dict) -> PhysicsSummary:
    wf = clip_dict.get("wave_features", {})
    quality = wf.get("quality", {})
    energy = wf.get("energy", {})
    damping = wf.get("damping", {})
    freq = wf.get("frequency", {})
    harmonic = wf.get("harmonic", {})

    return PhysicsSummary(
        exercise=clip_dict.get("exercise", ""),
        fps=clip_dict.get("fps", 30.0),
        n_frames=clip_dict.get("n_frames", 0),
        expert=clip_dict.get("expert", False),
        quality_grade=quality.get("grade", ""),
        smoothness=quality.get("smoothness", 0.0),
        control=quality.get("control", 0.0),
        efficiency_pct=energy.get("efficiency_pct", 0.0),
        consistency=quality.get("consistency", 0.0),
        damping_ratio=damping.get("ratio", 0.0),
        dominant_hz=freq.get("dominant_hz", 0.0),
        rep_count=harmonic.get("oscillation_count", 0),
        peak_power_w=energy.get("peak_power_w", 0.0),
        wave_phases=wf.get("waves", []),
        error_rate=clip_dict.get("error_rate", []),
    )


def build_llm_prompt(summary: PhysicsSummary, language_hint: str = "") -> str:
    system = (
        "You are a biomechanics-aware fitness coach. "
        "Given the physics analysis of an exercise clip, provide specific coaching feedback."
    )
    user_block = summary.to_prompt()
    if language_hint:
        user_block += f"\n\nPrior analysis: {language_hint[:500]}"
    user_block += "\n\nProvide coaching feedback:"
    return f"<|system|>\n{system}\n<|user|>\n{user_block}\n<|assistant|>\n"


def clip_to_prompt(clip_path: str) -> str:
    with open(clip_path, "r") as f:
        raw = json.load(f)
    summary = extract_physics_summary(raw)
    language = raw.get("LANGUAGE", "")
    return build_llm_prompt(summary, language)
