"""
Simplified Wave-Pose + LLM Training

End-to-end pipeline:
  2D Pose / Trajectories -> WavePose (Latent Encoder) -> WaveBridge -> LLM
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from MusclePose.data.loader import ClipDataset, TRAJECTORY_KEYS
from MusclePose.models.wavepose import WavePose, WaveOut
from MusclePose.wave_llm.bridge import extract_physics_summary, build_llm_prompt, WaveBridge

N_CHANNELS = len(TRAJECTORY_KEYS)   # 7 trajectory signals

def parse_args():
    p = argparse.ArgumentParser("Simplified Wave-Pose + LLM training")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)

    # LLM
    p.add_argument("--llm_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--llm_dim", type=int, default=3072)
    p.add_argument("--n_soft_tokens", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--max_text_len", type=int, default=256)

    return p.parse_args()

def load_llm_with_lora(model_name, lora_r, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
    )

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r*2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], # smaller lora for simplicity
        bias="none", task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_cfg)
    return llm, tokenizer

def compute_lm_loss(llm, tokenizer, bridge, wave_out, prompts, coaching_texts, device, max_len=256):
    soft_tokens = bridge(wave_out)
    B = soft_tokens.shape[0]

    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    target_enc = tokenizer(coaching_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)

    prompt_embeds = llm.get_input_embeddings()(prompt_enc["input_ids"])
    target_embeds = llm.get_input_embeddings()(target_enc["input_ids"])

    combined_embeds = torch.cat([soft_tokens.to(prompt_embeds.dtype), prompt_embeds, target_embeds], dim=1)
    n_prefix = soft_tokens.shape[1] + prompt_embeds.shape[1]

    ignore = torch.full((B, n_prefix), -100, dtype=torch.long, device=device)
    target_labels = target_enc["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        target_labels[target_labels == tokenizer.pad_token_id] = -100
    labels = torch.cat([ignore, target_labels], dim=1)

    soft_mask = torch.ones(B, soft_tokens.shape[1], device=device, dtype=prompt_enc["attention_mask"].dtype)
    attn_mask = torch.cat([soft_mask, prompt_enc["attention_mask"], target_enc["attention_mask"]], dim=1)

    out = llm(inputs_embeds=combined_embeds, attention_mask=attn_mask)
    shift_logits = out.logits[..., :-1, :].float()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    return loss

def collate_fn(batch):
    waves = torch.stack([b["waves"] for b in batch])
    languages     = [b["language"] for b in batch]
    exercises     = [b["exercise"] for b in batch]
    wave_features = [b["wave_features"] for b in batch]
    return {
        "waves": waves,
        "language": languages,
        "exercise": exercises,
        "wave_features": wave_features,
    }

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    ds = ClipDataset(args.data_dir)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, collate_fn=collate_fn)

    model = WavePose(d_in=N_CHANNELS).to(device)
    bridge = WaveBridge(llm_dim=args.llm_dim).to(device)
    llm, tokenizer = load_llm_with_lora(args.llm_name, args.lora_r, device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(bridge.parameters()) + [p for p in llm.parameters() if p.requires_grad],
        lr=args.lr
    )

    for epoch in range(1, args.epochs + 1):
        model.train(); bridge.train()
        for i, batch in enumerate(train_loader):
            tokens = batch["waves"].transpose(1, 2).to(device)
            out = model(tokens)
            
            prompts = []
            for wf, ex, lang in zip(batch["wave_features"], batch["exercise"], batch["language"]):
                summary = extract_physics_summary({"exercise": ex, "wave_features": wf})
                prompts.append(build_llm_prompt(summary, lang))
            
            loss = compute_lm_loss(llm, tokenizer, bridge, out, prompts, batch["language"], device, args.max_text_len)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % args.log_every == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] loss={loss.item():.4f}")

        # Simple validation
        model.eval(); bridge.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["waves"].transpose(1, 2).to(device)
                out = model(tokens)
                prompts = [build_llm_prompt(extract_physics_summary({"exercise": ex, "wave_features": wf}), lang)
                           for wf, ex, lang in zip(batch["wave_features"], batch["exercise"], batch["language"])]
                val_loss += compute_lm_loss(llm, tokenizer, bridge, out, prompts, batch["language"], device, args.max_text_len).item()
        print(f"Epoch {epoch} Validation loss={val_loss/len(val_loader):.4f}")

        torch.save({"model": model.state_dict(), "bridge": bridge.state_dict()}, os.path.join(args.save_dir, "latest.pt"))

if __name__ == "__main__":
    main()
