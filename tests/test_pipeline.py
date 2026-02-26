"""
Simplified WavePose Pipeline Tests
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
import numpy as np

def test_wavepose_forward():
    from MusclePose.models.wavepose import WavePose
    B, T, D = 2, 30, 7
    model = WavePose(d_in=D)
    tokens = torch.randn(B, T, D)
    out = model(tokens)
    assert out.phi.shape == (B, T, 256)
    assert out.dynamic_state.shape == (B, T, 32)
    print("  wavepose forward OK")

def test_wave_bridge():
    from MusclePose.models.wavepose import WaveOut
    from MusclePose.wave_llm.bridge import WaveBridge
    B, T = 2, 30
    fake_out = WaveOut(
        phi=torch.randn(B, T, 256),
        dynamic_state=torch.randn(B, T, 32)
    )
    bridge = WaveBridge(llm_dim=512, n_tokens=8)
    soft_tokens = bridge(fake_out)
    assert soft_tokens.shape == (B, 8, 512)
    print("  wave bridge OK")

def test_extract_summary():
    from MusclePose.wave_llm.bridge import extract_physics_summary
    clip_dict = {
        "exercise": "squat",
        "wave_features": {
            "quality": {"grade": "A", "smoothness": 0.95},
            "harmonic": {"oscillation_count": 5}
        }
    }
    summary = extract_physics_summary(clip_dict)
    assert summary.exercise == "squat"
    assert summary.quality_grade == "A"
    assert summary.rep_count == 5
    print("  extract summary OK")

if __name__ == "__main__":
    test_wavepose_forward()
    test_wave_bridge()
    test_extract_summary()
    print("\nAll simplified tests passed!")
