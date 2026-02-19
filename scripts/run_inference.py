import argparse
import numpy as np
import torch

from data.tokens import make_tokens_from_coco
from models.musclepose import MusclePoseCOCO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = np.load(args.kpts)
    kxy = torch.from_numpy(data["kxy"]).float()[None, ...]      # (1,T,17,2)
    kconf = torch.from_numpy(data["kconf"]).float()[None, ...]  # (1,T,17)

    tokens = make_tokens_from_coco(kxy, kconf)                  # (1,T,D)
    D = tokens.shape[-1]

    model = MusclePoseCOCO(d_in=D).to(args.device)
    model.eval()

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)

    with torch.no_grad():
        out = model(tokens.to(args.device))

    # save
    np.savez_compressed(
        args.out,
        q=out.q.cpu().numpy()[0],
        qdot=out.qdot.cpu().numpy()[0],
        qddot=out.qddot.cpu().numpy()[0],
        contact=out.contact.cpu().numpy()[0],
        tau_q=out.tau_q.cpu().numpy()[0],
        tau_mtg=out.tau_mtg.cpu().numpy()[0],
    )
    print("Saved:", args.out)

if __name__ == "__main__":
    main()