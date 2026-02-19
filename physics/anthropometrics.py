import torch
from dataclasses import dataclass

@dataclass
class AnthropometricsOut:
    mk: torch.Tensor     # (B,18)
    I0k: torch.Tensor    # (B,18,3,3)
    W: torch.Tensor      # (B,) body weight

def anthropometrics_from_beta_E(beta: torch.Tensor, E: torch.Tensor, device=None):
    """
    Minimal anthropometrics:
    - Estimate body mass M from beta (placeholder)
    - Segment mass distribution fixed + residual E
    - Inertia I0k ~ mk * Lk^2 * I
    """
    if device is None:
        device = beta.device
    B = beta.shape[0]
    K = 18

    # placeholder body mass: 70kg +/- small from beta
    M = 70.0 + 5.0 * torch.tanh(beta[:, 0])
    M = M.to(device)

    # base segment mass ratios (rough)
    sm = torch.tensor([
        0.14, 0.08, 0.08, 0.06,
        0.01, 0.03, 0.02, 0.01,
        0.01, 0.03, 0.02, 0.01,
        0.10, 0.05, 0.02,
        0.10, 0.05, 0.02
    ], device=device, dtype=beta.dtype)
    sm = sm / sm.sum()

    # E expected as (B,18) mass-ratio residuals (can be extended)
    if E.shape[-1] >= 18:
        sm = sm[None, :] + 0.02 * torch.tanh(E[:, :18])
        sm = sm / (sm.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        sm = sm[None, :].repeat(B, 1)

    mk = sm * M[:, None]

    # rough segment length scale
    Lk = torch.full((B, K), 0.3, device=device, dtype=beta.dtype)
    I0k = torch.zeros((B, K, 3, 3), device=device, dtype=beta.dtype)
    eye = torch.eye(3, device=device, dtype=beta.dtype)[None, None]
    I0k = mk[:, :, None, None] * (Lk[:, :, None, None] ** 2) * eye

    g = 9.8
    W = g * mk.sum(dim=-1)
    return AnthropometricsOut(mk=mk, I0k=I0k, W=W)