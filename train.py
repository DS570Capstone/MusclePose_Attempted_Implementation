"""
Unified MusclePose + LLM Training

End-to-end pipeline:
  COCO 2D keypoints -> MusclePose (physics) -> PhysicsBridge -> LLM coaching

Combined loss:
  L = L_physics + w_lm * L_lm

  L_physics = reproj + torque_consistency + smoothness + joint_limits + activation_sparsity
  L_lm      = causal language modelling on coaching text
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from MusclePose.data.tokens import make_tokens_from_coco
from MusclePose.models.musclepose import MusclePoseCOCO, PhysicsBridge, ForwardOut
from MusclePose.physics.skeleton47 import default_skeleton47, forward_kinematics

SKEL_TO_COCO = {
    0:  (11, 12),
    3:  (0,),
    5:  (5,),
    6:  (7,),
    7:  (9,),
    9:  (6,),
    10: (8,),
    11: (10,),
    12: (11,),
    13: (13,),
    14: (15,),
    15: (12,),
    16: (14,),
    17: (16,),
}


def parse_args():
    p = argparse.ArgumentParser("Unified MusclePose + LLM training")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_lm", type=float, default=2e-4)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--dt", type=float, default=1/30)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=10)

    # LLM
    p.add_argument("--llm_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--llm_dim", type=int, default=3072)
    p.add_argument("--n_soft_tokens", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--max_text_len", type=int, default=256)

    # loss weights
    p.add_argument("--w_reproj",  type=float, default=10.0)
    p.add_argument("--w_torque",  type=float, default=1.0)
    p.add_argument("--w_smooth",  type=float, default=0.1)
    p.add_argument("--w_jlim",   type=float, default=0.05)
    p.add_argument("--w_act",    type=float, default=0.01)
    p.add_argument("--w_lm",     type=float, default=1.0)
    return p.parse_args()


# ── Dataset ─────────────────────────────────────────────────────────────
class PairedDataset(Dataset):
    """
    Loads .pt files containing:
      kxy:           (T, 17, 2)
      kconf:         (T, 17)
      coaching_text: str  (optional, for LLM supervision)
    """
    def __init__(self, data_dir, seq_len=60):
        self.seq_len = seq_len
        self.samples = []
        for fn in sorted(os.listdir(data_dir)):
            if not fn.endswith(".pt"):
                continue
            d = torch.load(os.path.join(data_dir, fn), map_location="cpu", weights_only=True)
            kxy = d["kxy"]
            kconf = d["kconf"]
            text = d.get("coaching_text", "")
            T = kxy.shape[0]
            for start in range(0, T - seq_len + 1, seq_len // 2):
                self.samples.append((
                    kxy[start:start+seq_len],
                    kconf[start:start+seq_len],
                    text,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SyntheticPairedDataset(Dataset):
    """Generates random COCO keypoints + dummy coaching text for demo runs."""
    def __init__(self, n_samples=64, seq_len=60):
        self.n = n_samples
        self.seq_len = seq_len
        self.texts = [
            "Good form. Keep your back straight and maintain consistent tempo.",
            "Watch your knees tracking over toes. Slow down the eccentric phase.",
            "Excellent depth. Try engaging your core more at the bottom.",
        ]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        kxy = torch.randn(self.seq_len, 17, 2) * 100 + 300
        kconf = torch.ones(self.seq_len, 17)
        text = self.texts[idx % len(self.texts)]
        return kxy, kconf, text


# ── Losses ──────────────────────────────────────────────────────────────
def reprojection_loss(q, kxy_gt, kconf, skel):
    B, T, _ = q.shape
    q_flat = q.reshape(B * T, 47)
    _, p_joint, _ = forward_kinematics(q_flat, skel)
    p_joint = p_joint.reshape(B, T, 18, 3)
    proj_2d = p_joint[..., :2]
    loss = torch.tensor(0.0, device=q.device)
    n = 0
    for skel_idx, coco_ids in SKEL_TO_COCO.items():
        skel_pt = proj_2d[:, :, skel_idx]
        if len(coco_ids) == 2:
            target = 0.5 * (kxy_gt[:, :, coco_ids[0]] + kxy_gt[:, :, coco_ids[1]])
            conf = torch.min(kconf[:, :, coco_ids[0]], kconf[:, :, coco_ids[1]])
        else:
            target = kxy_gt[:, :, coco_ids[0]]
            conf = kconf[:, :, coco_ids[0]]
        diff = (skel_pt - target).pow(2).sum(dim=-1)
        loss = loss + (conf * diff).mean()
        n += 1
    return loss / max(n, 1)


def torque_loss(out: ForwardOut):
    return (out.tau_q[..., 6:] - out.tau_mtg).pow(2).mean()


def smoothness_loss(q):
    jerk = q[:, 3:] - 3*q[:, 2:-1] + 3*q[:, 1:-2] - q[:, :-3]
    return jerk.pow(2).mean()


def jlim_loss(q, qmin, qmax):
    qrot = q[..., 6:]
    return (torch.clamp(qmin - qrot, min=0).pow(2) + torch.clamp(qrot - qmax, min=0).pow(2)).mean()


def activation_loss(out: ForwardOut):
    return out.alpha.pow(2).mean()


# ── LLM helpers ─────────────────────────────────────────────────────────
def load_llm_with_lora(model_name, lora_r, lora_alpha, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    llm = prepare_model_for_kbit_training(llm)

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_cfg)
    llm.config.use_cache = False

    trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in llm.parameters())
    print(f"  LLM trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    return llm, tokenizer


def build_coaching_prompt(out: ForwardOut) -> str:
    """Build a text prompt from physics outputs (no clip JSON needed)."""
    q_range = out.q.detach().abs().mean().item()
    tau_mean = out.tau_q.detach().abs().mean().item()
    tau_mtg_mean = out.tau_mtg.detach().abs().mean().item()
    residual = (out.tau_q[..., 6:] - out.tau_mtg).detach().abs().mean().item()
    contact_pct = out.contact.detach().mean().item() * 100

    lines = [
        f"Joint angle range: {q_range:.3f} rad",
        f"ID torque mean: {tau_mean:.1f} Nm",
        f"MTG torque mean: {tau_mtg_mean:.1f} Nm",
        f"Torque residual: {residual:.2f} Nm",
        f"Ground contact: {contact_pct:.0f}%",
    ]
    return (
        "You are a biomechanics-aware fitness coach. "
        "Given the physics analysis below, provide coaching feedback.\n\n"
        + "\n".join(lines) + "\n\nCoaching feedback:"
    )


def compute_lm_loss(llm, tokenizer, bridge, phys_out, coaching_texts, device, max_len=256):
    """Forward through bridge + LLM, return language modelling loss."""
    soft_tokens = bridge(phys_out)
    B = soft_tokens.shape[0]

    prompts = ["Analyze the movement and provide coaching feedback:" for _ in range(B)]
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    target_enc = tokenizer(coaching_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)

    prompt_embeds = llm.get_input_embeddings()(prompt_enc["input_ids"])
    target_embeds = llm.get_input_embeddings()(target_enc["input_ids"])

    # [soft_physics_tokens, prompt_text_tokens, target_text_tokens]
    combined_embeds = torch.cat([soft_tokens.to(prompt_embeds.dtype), prompt_embeds, target_embeds], dim=1)
    n_prefix = soft_tokens.shape[1] + prompt_embeds.shape[1]

    # labels: -100 for soft+prompt tokens, actual ids for target tokens
    ignore = torch.full((B, n_prefix), -100, dtype=torch.long, device=device)
    labels = torch.cat([ignore, target_enc["input_ids"]], dim=1)

    # attention mask
    soft_mask = torch.ones(B, soft_tokens.shape[1], device=device)
    attn_mask = torch.cat([soft_mask, prompt_enc["attention_mask"], target_enc["attention_mask"]], dim=1)

    out = llm(inputs_embeds=combined_embeds, attention_mask=attn_mask, labels=labels)
    return out.loss


# ── Training loop ───────────────────────────────────────────────────────
def train_one_epoch(model, bridge, llm, tokenizer, loader, opt_phys, opt_lm, skel, args, epoch):
    model.train()
    bridge.train()
    device = args.device
    qmin, qmax = model.qmin, model.qmax

    stats = {k: 0.0 for k in ["total", "reproj", "torque", "smooth", "jlim", "act", "lm"]}
    n = 0

    for batch_idx, (kxy, kconf, texts) in enumerate(loader):
        kxy, kconf = kxy.to(device), kconf.to(device)
        tokens = make_tokens_from_coco(kxy, kconf)
        out = model(tokens)

        L_reproj = reprojection_loss(out.q, kxy, kconf, skel)
        L_torque = torque_loss(out)
        L_smooth = smoothness_loss(out.q)
        L_jlim   = jlim_loss(out.q, qmin, qmax)
        L_act    = activation_loss(out)

        L_physics = (args.w_reproj * L_reproj + args.w_torque * L_torque
                   + args.w_smooth * L_smooth + args.w_jlim * L_jlim + args.w_act * L_act)

        L_lm = compute_lm_loss(llm, tokenizer, bridge, out, list(texts), device, args.max_text_len)
        loss = L_physics + args.w_lm * L_lm

        opt_phys.zero_grad()
        opt_lm.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_phys.step()
        nn.utils.clip_grad_norm_(list(bridge.parameters()) + [p for p in llm.parameters() if p.requires_grad], 1.0)
        opt_lm.step()

        stats["total"]  += loss.item()
        stats["reproj"] += L_reproj.item()
        stats["torque"] += L_torque.item()
        stats["smooth"] += L_smooth.item()
        stats["jlim"]   += L_jlim.item()
        stats["act"]    += L_act.item()
        stats["lm"]     += L_lm.item()
        n += 1

        if (batch_idx + 1) % args.log_every == 0:
            avg = {k: v/n for k, v in stats.items()}
            print(f"  [{epoch}][{batch_idx+1}/{len(loader)}]  "
                  f"loss={avg['total']:.4f}  reproj={avg['reproj']:.4f}  "
                  f"torque={avg['torque']:.4f}  lm={avg['lm']:.4f}")

    return {k: v/max(n,1) for k, v in stats.items()}


@torch.no_grad()
def validate(model, bridge, llm, tokenizer, loader, skel, args):
    model.eval()
    bridge.eval()
    device = args.device
    total, n = 0.0, 0
    for kxy, kconf, texts in loader:
        kxy, kconf = kxy.to(device), kconf.to(device)
        tokens = make_tokens_from_coco(kxy, kconf)
        out = model(tokens)
        L_reproj = reprojection_loss(out.q, kxy, kconf, skel)
        L_torque = torque_loss(out)
        L_lm = compute_lm_loss(llm, tokenizer, bridge, out, list(texts), device, args.max_text_len)
        val = args.w_reproj * L_reproj + args.w_torque * L_torque + args.w_lm * L_lm
        total += val.item()
        n += 1
    return total / max(n, 1)


def collate_fn(batch):
    kxy = torch.stack([b[0] for b in batch])
    kconf = torch.stack([b[1] for b in batch])
    texts = [b[2] for b in batch]
    return kxy, kconf, texts


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("  MusclePose Unified Training")
    print("=" * 60)
    print(f"  device:  {device}")
    print(f"  epochs:  {args.epochs}")
    print(f"  llm:     {args.llm_name}")

    skel = default_skeleton47(device=device)

    # dataset
    ds = PairedDataset(args.data_dir, seq_len=args.seq_len)
    if len(ds) == 0:
        print("  No .pt data found, using synthetic demo data")
        ds = SyntheticPairedDataset(n_samples=64, seq_len=args.seq_len)

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"  train: {n_train}  val: {n_val}")

    # model (compute D_in from actual token output)
    dummy_kxy = torch.randn(1, args.seq_len, 17, 2)
    dummy_kconf = torch.ones(1, args.seq_len, 17)
    D_in = make_tokens_from_coco(dummy_kxy, dummy_kconf).shape[-1]
    model = MusclePoseCOCO(d_in=D_in, dt=args.dt).to(device)
    print(f"  MusclePose params: {sum(p.numel() for p in model.parameters()):,}")

    bridge = PhysicsBridge(d_model=256, llm_dim=args.llm_dim, n_tokens=args.n_soft_tokens).to(device)
    llm, tokenizer = load_llm_with_lora(args.llm_name, args.lora_r, args.lora_alpha, device)
    lm_params = list(bridge.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    opt_lm = torch.optim.AdamW(lm_params, lr=args.lr_lm, weight_decay=0.01)

    opt_phys = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phys, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, bridge, llm, tokenizer, train_loader, opt_phys, opt_lm, skel, args, epoch)
        val_loss = validate(model, bridge, llm, tokenizer, val_loader, skel, args)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}  train={train_stats['total']:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {"model": model.state_dict(), "bridge": bridge.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  -> saved best (val={best_val:.4f})")

        if epoch % args.save_every == 0:
            ckpt = {"epoch": epoch, "model": model.state_dict(), "bridge": bridge.state_dict(), "val_loss": val_loss}
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch{epoch}.pt"))

    adapter_path = os.path.join(args.save_dir, "lora_adapter")
    llm.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Saved LoRA adapter to {adapter_path}")

    print(f"\nDone. Best val={best_val:.4f}  Checkpoints in {args.save_dir}/")


if __name__ == "__main__":
    main()
