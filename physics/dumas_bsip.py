# musclepose/anthropometrics/dumas_bsip.py
from dataclasses import dataclass
import torch
import yaml

@dataclass
class SegmentBSIP:
    sm: float                 # mass fraction (m_k / M)
    com: list                 # [cx, cy, cz] as fraction of segment length in segment coords
    rg: list                  # [kxx, kyy, kzz] radius of gyration fractions (about segment axes)
    # Optional: products of inertia fractions if you have them:
    poi: list | None = None   # [kxy, kxz, kyz] as fractions (optional)

def load_dumas_yaml(path: str) -> dict[str, SegmentBSIP]:
    """
    YAML schema per segment:
      segment_name:
        sm: 0.1
        com: [0.4, 0.0, 0.0]
        rg:  [0.3, 0.2, 0.1]
        poi: [0.0, 0.0, 0.0]   # optional
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    out = {}
    for k, v in raw.items():
        out[k] = SegmentBSIP(
            sm=float(v["sm"]),
            com=[float(x) for x in v["com"]],
            rg=[float(x) for x in v["rg"]],
            poi=[float(x) for x in v["poi"]] if "poi" in v and v["poi"] is not None else None
        )
    return out

def inertia_from_bsip(mk: torch.Tensor, Lk: torch.Tensor, bsip: SegmentBSIP) -> torch.Tensor:
    """
    Build I0,k in segment coordinates from radii of gyration.
    mk: (...,)
    Lk: (...,)
    returns I0: (...,3,3)
    """
    # principal inertias: I = m (k*L)^2
    kxx, kyy, kzz = bsip.rg
    Ixx = mk * (kxx * Lk) ** 2
    Iyy = mk * (kyy * Lk) ** 2
    Izz = mk * (kzz * Lk) ** 2

    I = torch.zeros((*mk.shape, 3, 3), dtype=mk.dtype, device=mk.device)
    I[..., 0, 0] = Ixx
    I[..., 1, 1] = Iyy
    I[..., 2, 2] = Izz

    if bsip.poi is not None:
        kxy, kxz, kyz = bsip.poi
        Ixy = mk * kxy * Lk ** 2
        Ixz = mk * kxz * Lk ** 2
        Iyz = mk * kyz * Lk ** 2
        I[..., 0, 1] = I[..., 1, 0] = Ixy
        I[..., 0, 2] = I[..., 2, 0] = Ixz
        I[..., 1, 2] = I[..., 2, 1] = Iyz

    return I

def com_from_bsip(Lk: torch.Tensor, bsip: SegmentBSIP) -> torch.Tensor:
    """
    CoM location in segment coordinates (meters), scaled by segment length.
    Lk: (...,)
    returns com: (...,3)
    """
    frac = torch.tensor(bsip.com, dtype=Lk.dtype, device=Lk.device)  # (3,)
    return Lk[..., None] * frac