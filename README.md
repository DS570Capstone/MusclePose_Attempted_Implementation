# WavePose — Latent Wave Analysis for AI Coaching

A simplified 2D-pose to LLM pipeline that maps exercise trajectory signals into **latent wave embeddings**. These learned dynamic states are fed through a cross-attention bridge into an LLM (Phi-3.5-mini + LoRA) to generate rhythm-aware and technique-focused coaching feedback.

## Architecture

```
Clip JSON (7 trajectory signals × N frames)
        │
   ClipDataset                    normalise + pad/truncate → (C, T) waves
        │
   transpose → (T, 7)            each timestep = 7 body-region signals
        │
   WavePose (Motion Encoder)      8-layer transformer (d=256, 8 heads)
        │                         sinusoidal positional encoding
        ├──► phi (B, T, 256)      latent motion sequence
        └──► dynamic_state (B, T, 32) learned wave/dynamic analysis
        │
   WaveBridge (Cross-Attention)
        │  1. fuse: [phi | dynamic_state] → d_model (256)
        │  2. temporal positional encoding (sinusoidal)
        │  3. learned event tokens (8 queries, d_model)
        │  4. cross-attention (event tokens ← fused context)
        │  5. to_llm: d_model → llm_dim (3072)
        │  6. Gated scaling (tanh gate × 0.03) to match LLM embed scale
        │
        ▼  8 soft tokens (B, 8, 3072)
        │
   wave_features ─► PhysicsSummary ─► build_llm_prompt
        │           (quality grade, reps, wave phases)
        │
   [soft_tokens | prompt_embeds | target_embeds] ──► LLM input
        │
   LLM (Phi-3.5-mini-instruct + LoRA)
        ▼
   Coaching feedback text
```

### Key Features

- **Latent Wave Embeddings**: Replaces explicit physics equations with a learned `dynamic_state` head that captures the "wave" nature of exercise motion (rhythm, velocity, and phase).
- **Event-Based Bridge**: Uses learned temporal event tokens as cross-attention queries to identify key moments in the motion sequence (peak force, transitions, etc.) and encode them for the LLM.
- **Simplified Pipeline**: Removed all heavy physics dependencies (`inverse_dynamics`, `MTG`, `anthropometrics`) for a faster, more flexible latent-first approach.
- **LoRA Fine-tuning**: Efficiently adapts the LLM to understand motion embeddings using Low-Rank Adaptation.

## Folder Layout

| Path | Description |
|------|-------------|
| `models/wavepose.py` | `WavePose` encoder and `MotionEncoder` (Transformer) |
| `wave_llm/bridge.py` | `WaveBridge` (cross-attention) and prompt building logic |
| `data/loader.py` | `ClipDataset` — loads and normalises trajectory waves from JSON |
| `train.py` | End-to-end training of the encoder, bridge, and LoRA adapter |
| `infer.py` | Inference script: 2D-pose → Latent Wave Analysis → Coaching Text |
| `tests/test_pipeline.py` | Simplified unit tests for the wave-based pipeline |

## Usage

### 1. Training

Train the encoder and bridge on your exercise dataset:

```bash
python3 train.py \
  --data_dir data \
  --epochs 40 \
  --batch_size 4 \
  --lr 3e-4 \
  --llm_name microsoft/Phi-3.5-mini-instruct \
  --save_dir checkpoints
```

### 2. Inference

Generate coaching feedback from a motion clip:

```bash
python3 infer.py \
  --input data/clip_00002.json \
  --ckpt checkpoints/latest.pt \
  --adapter checkpoints/lora_adapter
```

### 3. Running Tests

```bash
python3 tests/test_pipeline.py
```

## Data Format

Each JSON clip should provide trajectory signals and metadata:
- `trajectory`, `legs_trajectory`, `core_`, etc. (7 signals total)
- `wave_features`: Pre-calculated metrics (reps, grade) used for prompt context.
- `LANGUAGE`: The ground-truth coaching text for training.
