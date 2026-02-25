# MusclePose — Physics-Informed LLM Coaching

Physics-informed neural network that estimates 47-DoF joint kinematics, inverse-dynamics torques, and Hill-type muscle torques from exercise trajectory signals — then feeds all of that through a cross-attention bridge into an LLM (Phi-3.5-mini + LoRA) that generates biomechanics-aware coaching feedback.

## Architecture

```
Clip JSON (7 trajectory signals × N frames)
        │
   ClipDataset                    normalise + pad/truncate → (C, T) waves
        │
   transpose → (T, 7)            each timestep = 7 body-region signals
        │
   MotionEncoder                  8-layer transformer (d=256, 8 heads)
        │                         sinusoidal positional encoding
        ├──► phi (B,T,256)        encoder hidden states
        ├──► q (47 DoF)           joint angles
        ├──► qdot, qddot          central finite differences
        ├──► beta, E              anthropometrics  ──► segment masses & inertias
        ├──► contact (L/R)        foot on/off ground (sigmoid threshold)
        ├──► delta                GRFM residuals   ──► ground reaction forces
        ├──► alpha (flex/ext)     muscle activations (82-d: flex + ext × 41 DoF)
        │
        ├──► inverse_dynamics     Newton–Euler via jacrev  ──► tau_q  (47)
        ├──► mtg_torque           Hill-type muscle model   ──► tau_mtg (41)
        │
   PhysicsBridge (cross-attention)
        │  1. physics_proj: concat(q, qdot, tau_q, tau_mtg, alpha, contact)
        │     → 266-d → d_model projection
        │  2. fuse: [phi | physics_proj] → d_model
        │  3. temporal positional encoding (sinusoidal)
        │  4. learned event tokens (8 queries, d_model)
        │  5. 2× cross-attention layers (event tokens ← fused context)
        │  6. to_llm: d_model → llm_dim (3072)
        │  7. LayerNorm + gated scaling (tanh gate × 0.03) to match LLM embed scale
        │
        ▼  8 soft tokens (B, 8, 3072)
        │
   wave_features ─► PhysicsSummary ─► build_llm_prompt
        │           (quality grade, reps, energy, damping, phases)
        │
   [soft_tokens | prompt_embeds | target_embeds] ──► LLM input
        │
   LLM (Phi-3.5-mini-instruct + LoRA)
        │  float32 cross-entropy loss (avoids bf16 underflow)
        │  supervised by LANGUAGE field from JSON
        ▼
   Coaching feedback text
```

### Key Design Decisions

- **PhysicsBridge** uses *learned temporal event tokens* as cross-attention queries over the full fused physics + encoder sequence. Each event token specialises on different temporal aspects (rep phases, transitions, peak-force moments), producing richer soft tokens than naive average pooling.
- **Gradient isolation**: the encoder is trained *only* by physics losses. All physics outputs (including `phi`) are detached before entering the bridge/LLM path. This prevents LM gradients from flowing back through jacrev-based inverse dynamics (which causes NaN gradients).
- **LM loss in float32**: the HF model normally computes cross-entropy in bfloat16, which underflows with a 32K vocab softmax. We extract logits and compute `F.cross_entropy` after upcasting to float32.
- **Gated scaling**: soft tokens are normalised (LayerNorm) then scaled by `tanh(gate) × 0.03` where `gate` is a learned scalar initialised at 0.5. This matches the ~0.03 std of LLM text embeddings, preventing the soft tokens from overwhelming or being invisible to the LLM.
- **LM warmup**: epoch 1 is physics-only (LM loss skipped) to let the encoder produce meaningful representations before the bridge starts training.
- **Torque loss detachment**: `tau_q` (from inverse dynamics) is treated as a teacher signal — detached in the loss — so gradients flow only through `tau_mtg` (muscle model path). Log-scale Smooth-L1 handles the magnitude mismatch between ID torques (~10⁴) and MTG torques (~10¹).
- **Padding masking**: padding tokens in the target coaching text are masked with `-100` so they don't contribute to the LM loss.

## Folder Layout

| Path | What it does |
|------|-------------|
| `models/musclepose.py` | `MusclePoseCOCO` (transformer encoder + all physics heads), `PhysicsBridge` (cross-attention bridge), `ForwardOut` dataclass |
| `physics/skeleton47.py` | 18-segment skeleton, forward kinematics (functional, jacrev-safe) |
| `physics/inverse_dynamics_autograd.py` | Newton–Euler inverse dynamics with automatic Jacobians via `torch.func.jacrev` |
| `physics/anthropometrics.py` | Segment masses and inertias from shape parameters (β, E) |
| `physics/grfm.py` | Ground reaction force model (linear spring + learned residuals δ) |
| `physics/mtg.py` | Hill-type muscle-tendon-group torque model |
| `physics/euler_jacobian.py` | ZXY Euler angle Jacobian and its time derivative |
| `physics/dumas_bsip.py` | Dumas body-segment inertial parameters from radii of gyration |
| `data/loader.py` | `ClipDataset` — loads clip JSON files, extracts 7-ch trajectory waves |
| `data/tokens.py` | COCO-17 keypoint tokeniser (legacy, used by unit tests only) |
| `wave_llm/bridge.py` | `PhysicsSummary`, `extract_physics_summary`, `build_llm_prompt` — structured prompt from clip JSON |
| `utils/rot.py` | Rotation helpers: `euler_ZXY_to_matrix`, `skew`, `rot_x/y/z` |
| `configs/dumas_bsip.yaml` | BSIP radii-of-gyration config |
| `train.py` | Unified training: physics losses + LM loss, dual optimisers, LM warmup, cosine scheduling |
| `infer.py` | Unified inference: physics → bridge → LLM generation with LoRA adapter |
| `tests/test_pipeline.py` | Unit tests for every module |

## Quick Start

### Environment Setup

```bash
module load cuda-12.9.0-gcc-12.1.0
module load mamba/latest
mamba activate wave_llm     # or your environment with torch, transformers, peft
```

### Run Tests

```bash
cd /scratch/jnolas77/fitness
python -m MusclePose.tests.test_pipeline
```

### Train (Physics + LLM End-to-End)

```bash
cd /scratch/jnolas77/fitness
python -m MusclePose.train \
  --data_dir MusclePose/data/Test-Data \
  --epochs 15 \
  --batch_size 2 \
  --seq_len 60 \
  --lr 3e-4 \
  --lr_lm 2e-4 \
  --llm_name microsoft/Phi-3.5-mini-instruct \
  --llm_dim 3072 \
  --n_soft_tokens 8 \
  --lora_r 32 \
  --lora_alpha 64 \
  --save_dir MusclePose/checkpoints \
  --device cuda \
  --log_every 4 \
  --save_every 5 \
  --val_split 0.15
```

Reads every `.json` clip in `--data_dir`. Outputs:
- `checkpoints/best.pt` — best model + bridge weights (by validation loss)
- `checkpoints/lora_adapter/` — saved LoRA adapter + tokenizer

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Directory containing clip `.json` files |
| `--epochs` | `40` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--seq_len` | `300` | Sequence length (frames) — pad/truncate |
| `--lr` | `3e-4` | Physics encoder learning rate |
| `--lr_lm` | `2e-4` | LM (bridge + LoRA) learning rate |
| `--llm_name` | `microsoft/Phi-3.5-mini-instruct` | HuggingFace LLM model |
| `--llm_dim` | `3072` | LLM hidden dimension |
| `--n_soft_tokens` | `8` | Number of soft physics tokens for the bridge |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha |
| `--max_text_len` | `256` | Max coaching text token length |
| `--w_torque` | `1.0` | Torque consistency loss weight |
| `--w_smooth` | `0.1` | Smoothness (jerk) loss weight |
| `--w_jlim` | `0.05` | Joint-limit loss weight |
| `--w_act` | `0.01` | Activation sparsity loss weight |
| `--w_lm` | `1.0` | Language modelling loss weight |

### Inference

```bash
cd /scratch/jnolas77/fitness
python -m MusclePose.infer \
  --checkpoint MusclePose/checkpoints/best.pt \
  --adapter_path MusclePose/checkpoints/lora_adapter \
  --input MusclePose/data/Test-Data/clip_00004.json \
  --seq_len 60 \
  --max_new_tokens 256 \
  --device cuda
```

Without `--input` it picks the first `.json` file found in `data/`.

**What happens:**
1. Reads a clip JSON → extracts 7 trajectory channels → prints clip summary (exercise, quality grade, reps, etc.)
2. Trajectory waves → MusclePose → prints physics outputs (joint angles, torques, muscle activations, contact)
3. PhysicsBridge encodes physics outputs into 8 soft tokens via cross-attention
4. `wave_features` → `extract_physics_summary` → `build_llm_prompt` → structured text prompt (Phi-3.5 chat format)
5. `[soft_tokens | prompt_embeds]` → LLM generates coaching feedback

### Data Format

Each JSON clip must contain:

| Field | Type | Description |
|-------|------|-------------|
| `trajectory` | `list[float]` | Main trajectory signal (N frames) |
| `legs_trajectory` | `list[float]` | Leg trajectory signal |
| `shoulder_trajectory` | `list[float]` | Shoulder trajectory signal |
| `back_trajectory` | `list[float]` | Back trajectory signal |
| `knee_angle_trajectory` | `list[float]` | Knee angle trajectory signal |
| `arm_Trajectory` | `list[float]` | Arm trajectory signal |
| `core_` | `list[float]` | Core trajectory signal |
| `wave_features` | `dict` | Contains `quality`, `energy`, `damping`, `frequency`, `harmonic`, `waves` sub-dicts |
| `LANGUAGE` | `str` | Ground-truth coaching text (training supervision target) |
| `exercise` | `str` | Exercise type (e.g. `"overhead_press"`, `"squat"`) |
| `fps` | `float` | Frames per second |
| `n_frames` | `int` | Total frame count |
| `expert` | `bool` | Whether the subject is an expert |
| `error_rate` | `list` | Error annotations |
| `video_id` | `str` | Clip identifier |

## Training Results

Verified training run (20 clips, 15 epochs, batch_size=2, seq_len=60):

```
Model parameters:  6,694,998 (MusclePose encoder + physics heads)
LLM trainable:     17,825,792 / 3,838,905,344 (0.46% LoRA)
Train/val split:   17 / 3 clips
```

| Epoch | Train Loss | Torque | LM Loss | Val Loss | Notes |
|------:|----------:|-------:|--------:|---------:|-------|
| 1 | 3.76 | 3.75 | 0.00 | 4.39 | Physics-only warmup (LM skipped) |
| 2 | 6.56 | 3.72 | 2.83 | 3.27 | LM starts, high initial loss |
| 4 | 4.72 | 3.71 | 1.01 | **2.78** | Best validation |
| 8 | 3.95 | 3.68 | 0.26 | 3.79 | LM converging |
| 15 | 3.49 | 3.45 | 0.03 | 4.46 | LM fully converged |

- **Zero NaN steps** across all 15 epochs
- LM loss: 2.83 → 0.03 (converged)
- Torque loss: 3.75 → 3.45 (steady improvement)
- Best validation: 2.78 (epoch 4)
- ~170 s/epoch on a single GPU

### Example Inference Output

**Input**: `clip_00004.json` — overhead press, non-expert, quality grade C

**Physics**:
```
  q          (1, 60, 47)   mean=0.0102
  tau_q      (1, 60, 47)   mean=17.62 Nm
  tau_mtg    (1, 60, 41)   mean=4.65 Nm
  torque residual: 3.9154 Nm
```

**Generated coaching**:
> The very low control and efficiency scores (0.30 and 0.05) indicate significant instability and wasted energy during the overhead press from your arm trajectory analysis. This suggests an unstable bar path or asymmetric drive between your left and right arms, leading to inefficient power transfer. Focus on maintaining a smooth, even vertical ascent while ensuring both arms move synchronously to improve control and efficiency. From the front view, the zero damping ratio and undamped eccentric phase further highlight the lack of controlled deceleration and potential instability during the lowering of the weight. Overall, prioritize driving symmetry and a smooth, controlled descent to improve form and prevent injury.

## Training Losses

| Loss | Weight | What it enforces |
|------|--------|-----------------|
| `L_torque` | 1.0 | `tau_q[6:] ≈ tau_mtg` — inverse dynamics ≈ muscle model (log-scale Smooth-L1, tau_q detached as teacher) |
| `L_smooth` | 0.1 | Low jerk in joint angles (3rd-order finite difference) |
| `L_jlim` | 0.05 | Joint angles stay within anatomical limits |
| `L_act` | 0.01 | Sparse muscle activations (L2 on alpha) |
| `L_lm` | 1.0 | Causal language modelling on coaching text (float32 cross-entropy, skipped epoch 1) |

**Optimisers**: Two separate AdamW optimisers with cosine annealing:
- Physics: `lr=3e-4`, `weight_decay=1e-4`
- LM (bridge + LoRA): `lr=2e-4`, `weight_decay=0.01`

## Assumptions

- Y-up right-handed coordinate system
- ZXY intrinsic Euler angles (ISB: z=flex/ext, x=abd/add, y=int/ext)
- 18 segments, 47 DoF (6 root + 41 rotational)
- 30 fps default (`dt = 1/30`)
- Gravity `[0, -9.81, 0]` m/s²
- SI units throughout (m, kg, s, N, Nm)
- LLM: Phi-3.5-mini-instruct with LoRA (r=32, alpha=64, targets: q/k/v/o/gate/up/down projections)

## Dependencies

- Python 3.10+
- PyTorch >= 2.0 (`torch.func.jacrev`)
- transformers >= 4.36
- peft >= 0.7
- NumPy, PyYAML
