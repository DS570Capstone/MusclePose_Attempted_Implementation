import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.tokens import make_tokens_from_coco
from models.musclepose import MusclePoseCOCO

class NPZDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        self.kxy = d["kxy"].astype(np.float32)     # (N,T,17,2)
        self.kconf = d["kconf"].astype(np.float32) # (N,T,17)
        self.q = d["q_gt"].astype(np.float32)      # (N,T,47)

    def __len__(self):
        return self.kxy.shape[0]

    def __getitem__(self, i):
        return self.kxy[i], self.kconf[i], self.q[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--out", default="outputs/ckpt.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tr = DataLoader(NPZDataset(args.train_npz), batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(NPZDataset(args.val_npz), batch_size=args.bs, shuffle=False, num_workers=2)

    # infer token dim
    kxy0, kc0, _ = next(iter(tr))
    tokens0 = make_tokens_from_coco(torch.from_numpy(kxy0.numpy()), torch.from_numpy(kc0.numpy()))
    D = tokens0.shape[-1]

    model = MusclePoseCOCO(d_in=D).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def run(loader, train=True):
        model.train(train)
        total = 0.0
        n = 0
        for kxy, kconf, q_gt in tqdm(loader, leave=False):
            kxy = torch.from_numpy(kxy.numpy()).to(args.device)
            kconf = torch.from_numpy(kconf.numpy()).to(args.device)
            q_gt = torch.from_numpy(q_gt.numpy()).to(args.device)

            tokens = make_tokens_from_coco(kxy, kconf)
            out = model(tokens)

            # losses
            Lq = torch.mean(torch.abs(out.q - q_gt))
            Lroot = torch.mean(torch.abs(out.tau_q[..., :6]))
            Ltau = torch.mean(torch.abs(out.tau_q[..., 6:] - out.tau_mtg))

            loss = Lq + 0.1*Lroot + 0.1*Ltau

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total += float(loss.item())
            n += 1
        return total / max(n, 1)

    best = 1e9
    for ep in range(1, args.epochs+1):
        tr_loss = run(tr, train=True)
        va_loss = run(va, train=False)
        print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "epoch": ep, "val": va_loss}, args.out)
            print("saved:", args.out)

if __name__ == "__main__":
    main()