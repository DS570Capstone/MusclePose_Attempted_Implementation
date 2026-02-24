"""
End-to-end tests for MusclePose pipeline.
Run: python -m tests.test_pipeline   (from parent of MusclePose)
  or: pytest tests/test_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLIP_PATH = os.path.join(DATA_DIR, "clip_00002.json")


def test_rotation_utils():
    from MusclePose.utils.rot import euler_ZXY_to_matrix, skew, rot_x, rot_y, rot_z
    angles = torch.randn(2, 3)
    R = euler_ZXY_to_matrix(angles)
    assert R.shape == (2, 3, 3), f"bad shape {R.shape}"
    det = torch.det(R)
    assert torch.allclose(det, torch.ones(2), atol=1e-5), f"not rotation: det={det}"
    RtR = R.transpose(-1, -2) @ R
    assert torch.allclose(RtR, torch.eye(3).expand(2, -1, -1), atol=1e-5)
    v = torch.randn(3)
    S = skew(v)
    assert S.shape == (3, 3)
    assert torch.allclose(S + S.T, torch.zeros(3, 3), atol=1e-7)
    print("  rotation utils OK")


def test_skeleton_fk():
    from MusclePose.physics.skeleton47 import default_skeleton47, forward_kinematics, unpack_q_to_local_angles
    skel = default_skeleton47()
    assert len(skel.names) == 18
    assert sum(skel.dof) == 44  # 41 rotational + 3 root rot
    q = torch.zeros(1, 47)
    R0k, p_joint, p_com = forward_kinematics(q, skel)
    assert R0k.shape == (1, 18, 3, 3)
    assert p_joint.shape == (1, 18, 3)
    assert p_com.shape == (1, 18, 3)
    root_pos, angles = unpack_q_to_local_angles(q, skel)
    assert root_pos.shape == (1, 3)
    assert angles.shape == (1, 18, 3)
    print("  skeleton FK OK")


def test_euler_jacobian():
    from MusclePose.physics.euler_jacobian import jomega_zxy, djomega_zxy
    theta = torch.randn(3)
    J = jomega_zxy(theta)
    assert J.shape == (3, 3)
    theta_dot = torch.randn(3)
    dJ = djomega_zxy(theta, theta_dot)
    assert dJ.shape == (3, 3)
    eps = 1e-5
    theta2 = theta + eps * theta_dot
    J2 = jomega_zxy(theta2)
    dJ_fd = (J2 - J) / eps
    assert torch.allclose(dJ, dJ_fd, atol=1e-2), f"dJ mismatch: max err={torch.abs(dJ - dJ_fd).max():.6f}"
    print("  euler jacobian OK")


def test_anthropometrics():
    from MusclePose.physics.anthropometrics import anthropometrics_from_beta_E
    B = 2
    beta = torch.randn(B, 10)
    E = torch.randn(B, 18)
    out = anthropometrics_from_beta_E(beta, E)
    assert out.mk.shape == (B, 18)
    assert out.I0k.shape == (B, 18, 3, 3)
    assert out.W.shape == (B,)
    assert (out.mk > 0).all(), "segment masses must be positive"
    assert (out.W > 0).all(), "body weight must be positive"
    print("  anthropometrics OK")


def test_grfm():
    from MusclePose.physics.grfm import grfm_from_kinematics
    B, T = 1, 4
    Psi = torch.randn(B, T, 2, 8)
    delta = torch.randn(B, T, 2, 7)
    contact = torch.ones(B, T, 2)
    bw = torch.tensor([700.0])
    foot_geom = torch.tensor([0.25, 0.10, 0.05]).expand(B, 2, 3).clone()
    R0 = torch.eye(3).expand(B, T, 2, 3, 3).clone()
    out = grfm_from_kinematics(Psi, delta, contact, bw, foot_geom, R0)
    assert out.F.shape == (B, T, 2, 3)
    assert out.M.shape == (B, T, 2, 3)
    assert out.mu.shape == (B, T, 2)
    print("  GRFM OK")


def test_mtg():
    from MusclePose.physics.mtg import mtg_torque
    B, T, D = 1, 4, 3
    q = torch.randn(B, T, D)
    qdot = torch.randn(B, T, D)
    af = torch.sigmoid(torch.randn(B, T, D))
    ae = torch.sigmoid(torch.randn(B, T, D))
    tau0 = 100 * torch.ones(B, T, D)
    wmax = 10 * torch.ones(B, T, D)
    qmin = -torch.ones(B, T, D)
    qmax = torch.ones(B, T, D)
    ga = torch.tensor([1.2, 0.2, 0.3, 1.0, 0.0, -0.1]).expand(B, T, D, 6).clone()
    gp = torch.tensor([0.5, 1.0, 0.4, 1.0, 0.05]).expand(B, T, D, 5).clone()
    tau = mtg_torque(q, qdot, af, ae, tau0, wmax, qmin, qmax, ga, gp)
    assert tau.shape == (B, T, D)
    print("  MTG OK")


def test_finite_differences():
    from MusclePose.models.musclepose import finite_differences
    B, T, D = 2, 10, 3
    x = torch.randn(B, T, D)
    xdot, xddot = finite_differences(x, dt=1 / 30)
    assert xdot.shape == (B, T, D)
    assert xddot.shape == (B, T, D)
    print("  finite differences OK")


def test_physics_bridge():
    from MusclePose.models.musclepose import PhysicsBridge, ForwardOut
    B, T = 2, 30
    fake_out = ForwardOut(
        q=torch.randn(B, T, 47),
        qdot=torch.randn(B, T, 47),
        qddot=torch.randn(B, T, 47),
        contact=torch.ones(B, T, 2),
        E=torch.randn(B, 18),
        delta=torch.randn(B, T, 2, 7),
        Z=torch.randn(B, 41),
        alpha=torch.randn(B, T, 82),
        tau_q=torch.randn(B, T, 47),
        tau_mtg=torch.randn(B, T, 41),
        phi=torch.randn(B, T, 256),
    )
    bridge = PhysicsBridge(d_model=256, llm_dim=512, n_tokens=8)
    tokens = bridge(fake_out)
    assert tokens.shape == (B, 8, 512), f"bad shape {tokens.shape}"
    print("  physics bridge OK")


def test_clip_loader():
    from MusclePose.data.loader import ClipDataset
    if not os.path.exists(CLIP_PATH):
        print("  SKIP clip loader (no data file)")
        return
    ds = ClipDataset(DATA_DIR, sequence_length=300)
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["waves"].shape[0] == 7  # 7 trajectory channels
    assert sample["waves"].shape[1] == 300
    assert sample["mask"].shape == (300,)
    assert isinstance(sample["language"], str)
    assert isinstance(sample["wave_features"], dict)
    print(f"  clip loader OK — loaded '{sample['video_id']}', exercise={sample['exercise']}")


def test_physics_summary_bridge():
    from MusclePose.wave_llm.bridge import extract_physics_summary, build_llm_prompt
    if not os.path.exists(CLIP_PATH):
        print("  SKIP physics bridge (no data file)")
        return
    with open(CLIP_PATH) as f:
        raw = json.load(f)
    summary = extract_physics_summary(raw)
    assert summary.exercise == "squat"
    assert summary.quality_grade == "D"
    assert summary.rep_count > 0
    prompt = build_llm_prompt(summary, raw.get("LANGUAGE", ""))
    assert "Exercise: squat" in prompt
    assert "<|assistant|>" in prompt
    print(f"  physics bridge OK — grade={summary.quality_grade}, reps={summary.rep_count}")


def test_end_to_end_unified():
    """Full pipeline: JSON waves -> MusclePose -> PhysicsBridge -> soft tokens."""
    from MusclePose.models.musclepose import MusclePoseCOCO, PhysicsBridge
    from MusclePose.data.loader import TRAJECTORY_KEYS
    B, T, C = 1, 30, len(TRAJECTORY_KEYS)   # 7 channels
    waves = torch.rand(B, C, T)              # (B, C, T)  like ClipDataset output
    tokens = waves.transpose(1, 2)           # (B, T, 7)
    model = MusclePoseCOCO(d_in=C, dt=1/30)
    bridge = PhysicsBridge(d_model=256, llm_dim=512, n_tokens=8)
    model.eval()
    bridge.eval()
    with torch.no_grad():
        out = model(tokens)
        soft_tokens = bridge(out)
    assert out.phi.shape == (B, T, 256)
    assert soft_tokens.shape == (B, 8, 512)
    assert out.tau_q.shape[0] == B
    assert out.q.shape == (B, T, 47)
    print(f"  end-to-end unified OK — waves({C}ch) -> physics + bridge")


def test_clip_to_model():
    """Load the real JSON clip, run through ClipDataset -> MusclePose -> PhysicsBridge."""
    from MusclePose.data.loader import ClipDataset, TRAJECTORY_KEYS
    from MusclePose.models.musclepose import MusclePoseCOCO, PhysicsBridge
    from MusclePose.wave_llm.bridge import extract_physics_summary, build_llm_prompt
    if not os.path.exists(CLIP_PATH):
        print("  SKIP clip_to_model (no data file)")
        return
    ds = ClipDataset(DATA_DIR, sequence_length=300)
    sample = ds[0]
    waves = sample["waves"].unsqueeze(0)         # (1, 7, 300)
    tokens = waves.transpose(1, 2)               # (1, 300, 7)

    C = len(TRAJECTORY_KEYS)
    model = MusclePoseCOCO(d_in=C, dt=1/30)
    bridge = PhysicsBridge(d_model=256, llm_dim=512, n_tokens=8)
    model.eval()
    bridge.eval()
    with torch.no_grad():
        out = model(tokens)
        soft_tokens = bridge(out)

    assert out.q.shape == (1, 300, 47)
    assert soft_tokens.shape == (1, 8, 512)

    # also verify wave_features -> prompt works
    clip_dict = {"exercise": sample["exercise"], "wave_features": sample["wave_features"]}
    summary = extract_physics_summary(clip_dict)
    prompt = build_llm_prompt(summary, sample["language"])
    assert "Exercise:" in prompt
    assert "<|assistant|>" in prompt

    print(f"  clip_to_model OK — {sample['video_id']}, exercise={sample['exercise']}, "
          f"q={out.q.shape}, soft_tokens={soft_tokens.shape}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    print("Running MusclePose tests\n")
    tests = [
        test_rotation_utils,
        test_skeleton_fk,
        test_euler_jacobian,
        test_anthropometrics,
        test_grfm,
        test_mtg,
        test_finite_differences,
        test_physics_bridge,
        test_clip_loader,
        test_physics_summary_bridge,
        test_end_to_end_unified,
        test_clip_to_model,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")
    sys.exit(1 if failed else 0)
