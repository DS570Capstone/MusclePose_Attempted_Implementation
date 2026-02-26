"""
Simplified WavePose Inference
"""
import argparse
import json
import torch
from MusclePose.models.wavepose import WavePose
from MusclePose.wave_llm.bridge import WaveBridge, extract_physics_summary, build_llm_prompt
from MusclePose.data.loader import TRAJECTORY_KEYS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to clip JSON")
    p.add_argument("--ckpt", type=str, required=True, help="Path to wavepose checkpoint")
    p.add_argument("--llm", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load data
    with open(args.input, "r") as f:
        raw = json.load(f)
    
    waves = []
    for k in TRAJECTORY_KEYS:
        signal = torch.tensor(raw.get(k, []), dtype=torch.float32)
        if len(signal) == 0: signal = torch.zeros(300)
        # norm
        if signal.max() - signal.min() > 1e-8:
            signal = (signal - signal.min()) / (signal.max() - signal.min())
        # pad/trunc
        if len(signal) > 300: signal = signal[:300]
        else:
            tmp = torch.zeros(300)
            tmp[:len(signal)] = signal
            signal = tmp
        waves.append(signal)
    tokens = torch.stack(waves).unsqueeze(0).transpose(1, 2).to(device) # (1, 300, 7)

    # Load models
    model = WavePose(d_in=len(TRAJECTORY_KEYS)).to(device)
    bridge = WaveBridge().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    bridge.load_state_dict(ckpt["bridge"])

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        llm = PeftModel.from_pretrained(llm, args.adapter)
    
    model.eval(); bridge.eval(); llm.eval()

    # Forward
    out = model(tokens)
    soft_tokens = bridge(out)

    summary = extract_physics_summary(raw)
    prompt = build_llm_prompt(summary)
    
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)
    
    inputs_embeds = torch.cat([soft_tokens.to(prompt_embeds.dtype), prompt_embeds], dim=1)
    
    gen = llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=128, do_sample=True, temperature=0.7)
    print("\n--- Coaching Feedback ---\n")
    print(tokenizer.decode(gen[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
