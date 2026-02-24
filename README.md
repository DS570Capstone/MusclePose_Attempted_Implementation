# MusclePose Attempted integrated with the LLM

Physics-informed neural network that estimates 47-DoF joint kinematics, inverse-dynamics torques, and Hill-type muscle torques from 2D COCO keypoints — then injects all of that into an LLM for coaching feedback.

## How It Works

```
Clip JSON (7 trajectory signals × N frames)
        │
   ClipDataset                    normalise + pad/truncate → (7, T) waves
        │
   transpose → (T, 7)            each timestep = 7 body-region signals
        │
   MotionEncoder                  8-layer transformer (d=256)
        │
        ├──► q (47 DoF)           joint angles
        ├──► qdot, qddot          finite differences
        ├──► beta, E              anthropometrics  ──► segment masses & inertias
        ├──► contact (L/R)        foot on/off ground
        ├──► delta                GRFM residuals   ──► ground reaction forces
        ├──► alpha (flex/ext)     muscle activations
        │
        ├──► inverse_dynamics     Newton–Euler via jacrev  ──► tau_q (47)
        ├──► mtg_torque           Hill-type muscle model   ──► tau_mtg (41)
        │
   PhysicsBridge                  encodes ALL physics outputs (q, qdot,
        │                         tau_q, tau_mtg, alpha, contact) + encoder
        │                         features into 8 soft tokens
        │
   wave_features ─► PhysicsSummary ─► build_llm_prompt (quality, reps, etc.)
        │
   LLM (Phi-3.5-mini + LoRA)     generates coaching text
        │                         supervised by LANGUAGE field from JSON
```

The key difference from a naive approach: **PhysicsBridge** doesn't just pass raw encoder features to the LLM. It takes the actual computed physics (torques, joint angles, muscle activations, ground contact) and projects them alongside the encoder hidden states into the LLM's embedding space. The text prompt is built from the clip's **wave_features** (quality grade, smoothness, energy, reps, etc.) via `extract_physics_summary` + `build_llm_prompt`. The LLM sees real biomechanics + structured movement analysis, and learns to produce coaching text supervised by the **LANGUAGE** field from each JSON clip.

## Folder Layout

| Path | What it does |
|------|-------------|
| `physics/skeleton47.py` | 18-segment skeleton, forward kinematics (functional, jacrev-safe) |
| `physics/inverse_dynamics_autograd.py` | Newton–Euler ID with automatic Jacobians via `torch.func.jacrev` |
| `physics/anthropometrics.py` | Segment masses and inertias from shape parameters |
| `physics/grfm.py` | Ground reaction force model (linear + learned residuals) |
| `physics/mtg.py` | Hill-type muscle-tendon-group torque model |
| `physics/euler_jacobian.py` | ZXY Euler angle Jacobian and its time derivative |
| `physics/dumas_bsip.py` | Dumas body-segment inertial parameters from radii of gyration |
| `models/musclepose.py` | `MusclePoseCOCO` (transformer + physics heads), `PhysicsBridge`, `ForwardOut` |
| `data/tokens.py` | COCO-17 keypoint tokeniser (legacy, used by unit tests only) |
| `data/loader.py` | `ClipDataset` — loads clip JSON files, extracts 7-ch trajectory waves |
| `wave_llm/bridge.py` | `PhysicsSummary` + `extract_physics_summary` + `build_llm_prompt` from clip JSON |
| `utils/rot.py` | Rotation helpers: `euler_ZXY_to_matrix`, `skew`, `rot_x/y/z` |
| `configs/dumas_bsip.yaml` | BSIP radii-of-gyration config |
| `train.py` | Unified training on JSON clips: physics losses + LM loss in one backward pass |
| `infer.py` | Unified inference on a clip JSON: physics → bridge → LLM generation |
| `tests/test_pipeline.py` | Unit tests for every module |

## Quick Start

### Run tests

```bash
cd <parent of MusclePose/>
python -m MusclePose.tests.test_pipeline
```

### Train (physics + LLM end-to-end)

```bash
python -m MusclePose.train \
  --data_dir MusclePose/data \
  --epochs 40 \
  --batch_size 4 \
  --llm_name microsoft/Phi-3.5-mini-instruct \
  --save_dir MusclePose/checkpoints
```

Reads every `.json` clip in `--data_dir`.

**Data format**: each JSON clip must contain:
- 7 trajectory signals: `trajectory`, `legs_trajectory`, `shoulder_trajectory`, `back_trajectory`, `knee_angle_trajectory`, `arm_Trajectory`, `core_` — each a list of N floats (one per frame)
- `wave_features` — `dict` with `quality`, `energy`, `damping`, `frequency`, `harmonic`, `waves` sub-dicts (used to build the LLM prompt)
- `LANGUAGE` — `str` ground-truth coaching text (supervision target for LM loss)
- `exercise` — `str` exercise type (e.g. "squat")
- `fps`, `n_frames`, `expert`, `error_rate`, `video_id` — metadata

### Inference

```bash
python -m MusclePose.infer \
  --checkpoint MusclePose/checkpoints/best.pt \
  --input MusclePose/data/clip_00002.json \
  --adapter_path MusclePose/checkpoints/lora_adapter
```

Without `--input` it picks the first `.json` file found in `data/`.

**What happens:**
1. Reads a clip JSON → extracts 7 trajectory channels → prints clip summary (exercise, quality grade, reps, etc.)
2. Trajectory waves → MusclePose → prints physics outputs (joint angles, torques, contact)
3. PhysicsBridge encodes physics outputs into 8 soft tokens
4. `wave_features` → `extract_physics_summary` → `build_llm_prompt` → structured text prompt
5. Soft tokens + text prompt → LLM generates coaching feedback

### Training losses

| Loss | Weight | What it enforces |
|------|--------|-----------------|
| `L_torque` | 1.0 | `tau_q[6:] ≈ tau_mtg` (Newton–Euler = muscle model) |
| `L_smooth` | 0.1 | Low jerk in joint angles |
| `L_jlim` | 0.05 | Stay within joint limits |
| `L_act` | 0.01 | Sparse muscle activations |
| `L_lm` | 1.0 | Causal language modelling on coaching text |

## Assumptions

- Y-up right-handed coordinate system
- ZXY intrinsic Euler angles (ISB: z=flex/ext, x=abd/add, y=int/ext)
- 18 segments, 47 DoF (6 root + 41 rotational)
- 30 fps default (`dt = 1/30`)
- Gravity `[0, -9.81, 0]` m/s²
- SI units throughout (m, kg, s, N, Nm)
- LLM: Phi-3.5-mini-instruct with LoRA (r=32, alpha=64)

## Dependencies

- PyTorch >= 2.0 (`torch.func.jacrev`)
- transformers, peft, bitsandbytes
- NumPy, PyYAML
