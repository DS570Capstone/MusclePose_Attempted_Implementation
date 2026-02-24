"""
Unified MusclePose + LLM Inference

COCO 2D keypoints -> MusclePose (physics) -> PhysicsBridge -> LLM coaching text
"""

import argparse
import os
import sys
import torch

from MusclePose.data.tokens import make_tokens_from_coco
from MusclePose.models.musclepose import MusclePoseCOCO, PhysicsBridge


def parse_args():
    p = argparse.ArgumentParser("Unified MusclePose + LLM inference")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--input", type=str, default=None, help=".pt file with kxy (T,17,2) + kconf (T,17)")
    p.add_argument("--llm_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--adapter_path", type=str, default="checkpoints/lora_adapter")
    p.add_argument("--llm_dim", type=int, default=3072)
    p.add_argument("--n_soft_tokens", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_input(path, device):
    if path is not None:
        d = torch.load(path, map_location=device, weights_only=True)
        kxy = d["kxy"].unsqueeze(0).to(device)
        kconf = d["kconf"].unsqueeze(0).to(device)
        return kxy, kconf

    T = 60
    print("  No input file, generating synthetic COCO keypoints")
    kxy = (torch.randn(1, T, 17, 2) * 100 + 300).to(device)
    kconf = torch.ones(1, T, 17, device=device)
    return kxy, kconf


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


def build_physics_prompt(out):
    q_range = out.q.abs().mean().item()
    tau_mean = out.tau_q.abs().mean().item()
    tau_mtg_mean = out.tau_mtg.abs().mean().item()
    residual = (out.tau_q[..., 6:] - out.tau_mtg).abs().mean().item()
    contact_pct = out.contact.mean().item() * 100

    return (
        "You are a biomechanics-aware fitness coach. "
        "Given the physics analysis below, provide specific coaching feedback.\n\n"
        f"Joint angle range: {q_range:.3f} rad\n"
        f"Inverse dynamics torque: {tau_mean:.1f} Nm\n"
        f"Muscle model torque: {tau_mtg_mean:.1f} Nm\n"
        f"Torque residual: {residual:.2f} Nm\n"
        f"Ground contact: {contact_pct:.0f}%\n\n"
        "Coaching feedback:"
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("  MusclePose Unified Inference")
    print("=" * 60)

    # load input
    kxy, kconf = load_input(args.input, device)
    tokens = make_tokens_from_coco(kxy, kconf)
    print(f"  Input: {kxy.shape[1]} frames, tokens {tokens.shape}")

    # load MusclePose
    D_in = tokens.shape[-1]
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

    # build text prompt
    prompt_text = build_physics_prompt(out)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(device)
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
