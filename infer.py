"""
Unified MusclePose + LLM Inference

Clip JSON -> trajectory waves -> MusclePose (physics) -> PhysicsBridge -> LLM coaching text

Reads the same JSON format used for training:
  trajectory, legs_trajectory, shoulder_trajectory, back_trajectory,
  knee_angle_trajectory, arm_Trajectory, core_  (7 x N_FRAMES floats)
  wave_features, LANGUAGE, exercise, fps, n_frames, ...
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

from MusclePose.data.loader import ClipDataset, TRAJECTORY_KEYS
from MusclePose.models.musclepose import MusclePoseCOCO, PhysicsBridge
from MusclePose.wave_llm.bridge import extract_physics_summary, build_llm_prompt


N_CHANNELS = len(TRAJECTORY_KEYS)  # 7


def parse_args():
    p = argparse.ArgumentParser("Unified MusclePose + LLM inference")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--input", type=str, default=None, help="path to a clip .json file")
    p.add_argument("--llm_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--adapter_path", type=str, default="checkpoints/lora_adapter")
    p.add_argument("--llm_dim", type=int, default=3072)
    p.add_argument("--n_soft_tokens", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_clip(path, seq_len=300):
    """Load a single clip JSON and return waves (1, C, T) + raw dict."""
    with open(path, "r") as f:
        raw = json.load(f)

    waves = []
    for key in TRAJECTORY_KEYS:
        signal = np.array(raw.get(key, []), dtype=np.float32)
        if len(signal) == 0:
            signal = np.zeros(seq_len, dtype=np.float32)
        # normalise
        lo, hi = signal.min(), signal.max()
        if hi - lo > 1e-8:
            signal = (signal - lo) / (hi - lo)
        # pad or truncate
        if len(signal) >= seq_len:
            signal = signal[:seq_len]
        else:
            padded = np.zeros(seq_len, dtype=np.float32)
            padded[:len(signal)] = signal
            signal = padded
        waves.append(torch.from_numpy(signal))

    stacked = torch.stack(waves, dim=0).unsqueeze(0)  # (1, C, T)
    return stacked, raw


def print_physics(out):
    print("\n--- Physics Outputs ---")
    print(f"  q          {out.q.shape}   mean={out.q.mean():.4f}")
    print(f"  qdot       {out.qdot.shape}   mean={out.qdot.mean():.4f}")
    print(f"  qddot      {out.qddot.shape}   mean={out.qddot.mean():.4f}")
    print(f"  contact    {out.contact.shape}   pct={out.contact.mean().item()*100:.0f}%")
    print(f"  tau_q      {out.tau_q.shape}   mean={out.tau_q.abs().mean():.2f} Nm")
    print(f"  tau_mtg    {out.tau_mtg.shape}   mean={out.tau_mtg.abs().mean():.2f} Nm")
    residual = (out.tau_q[..., 6:] - out.tau_mtg).abs().mean().item()
    print(f"  torque residual: {residual:.4f} Nm")
    print(f"  alpha      {out.alpha.shape}   mean={out.alpha.mean():.4f}")


def print_clip_summary(raw):
    """Print key info from the JSON clip."""
    print("\n--- Clip Info ---")
    print(f"  video_id:  {raw.get('video_id', '?')}")
    print(f"  exercise:  {raw.get('exercise', '?')}")
    print(f"  expert:    {raw.get('expert', '?')}")
    print(f"  fps:       {raw.get('fps', '?')}")
    print(f"  n_frames:  {raw.get('n_frames', '?')}")
    wf = raw.get("wave_features", {})
    q = wf.get("quality", {})
    e = wf.get("energy", {})
    h = wf.get("harmonic", {})
    print(f"  quality:   grade={q.get('grade','?')}  smoothness={q.get('smoothness',0):.2f}  "
          f"control={q.get('control',0):.2f}  consistency={q.get('consistency',0):.2f}")
    print(f"  energy:    efficiency={e.get('efficiency_pct',0):.1f}%  peak_power={e.get('peak_power_w',0):.1f} W")
    print(f"  reps:      {h.get('oscillation_count', '?')}")
    lang = raw.get("LANGUAGE", "")
    if lang:
        print(f"  LANGUAGE:  {lang[:120]}...")


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("  MusclePose Unified Inference (JSON clip)")
    print("=" * 60)

    if args.input is None:
        # default: look for first .json in data/
        import glob
        clips = sorted(glob.glob(os.path.join("data", "*.json")))
        if not clips:
            clips = sorted(glob.glob(os.path.join("MusclePose", "data", "*.json")))
        if not clips:
            print("  ERROR: No --input provided and no .json found in data/")
            sys.exit(1)
        args.input = clips[0]
        print(f"  No --input provided, using {args.input}")

    # load clip JSON
    waves, raw = load_clip(args.input, seq_len=args.seq_len)
    waves = waves.to(device)                                    # (1, C, T)
    tokens = waves.transpose(1, 2)                              # (1, T, C=7)
    print(f"  Input: {args.input}  ({raw.get('n_frames', '?')} frames, {N_CHANNELS} channels)")
    print_clip_summary(raw)

    # load MusclePose
    D_in = N_CHANNELS
    model = MusclePoseCOCO(d_in=D_in, dt=1/30).to(device)
    bridge = PhysicsBridge(d_model=256, llm_dim=args.llm_dim, n_tokens=args.n_soft_tokens).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "bridge" in ckpt:
            bridge.load_state_dict(ckpt["bridge"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  No checkpoint found at {args.checkpoint}, using random weights")

    model.eval()
    bridge.eval()

    # run MusclePose
    with torch.no_grad():
        out = model(tokens)
    print_physics(out)

    # load LLM
    print("\n--- Loading LLM ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(
            args.llm_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        llm = PeftModel.from_pretrained(base, args.adapter_path)
        llm = llm.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        print(f"  Loaded LoRA adapter from {args.adapter_path}")
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        print(f"  Using base model (no adapter found at {args.adapter_path})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.eval()

    # bridge physics -> soft tokens
    with torch.no_grad():
        soft_tokens = bridge(out).to(torch.bfloat16)

    # build text prompt from the clip's actual wave_features
    summary = extract_physics_summary(raw)
    prompt_text = build_llm_prompt(summary, raw.get("LANGUAGE", ""))
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_enc["input_ids"])

    # combine: [soft_physics_tokens | prompt_text_tokens]
    combined = torch.cat([soft_tokens, prompt_embeds], dim=1)
    attn_mask = torch.ones(1, combined.shape[1], device=device)

    # generate
    print("\n--- Generating coaching feedback ---\n")
    with torch.no_grad():
        gen_ids = llm.generate(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    print(response)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
