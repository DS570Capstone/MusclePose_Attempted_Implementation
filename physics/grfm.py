import torch
from dataclasses import dataclass

def build_psi_ankle(P_ankle_y, P_opp_ankle_y,
                    dP_ankle_y, ddP_ankle_y,
                    q_ankle_z, dq_ankle_z, ddq_ankle_z):
    """
    All inputs are (T,) or (B,T). Returns Psi with shape (T,8) or (B,T,8).
    """
    ones = torch.ones_like(P_ankle_y)
    Psi = torch.stack([
        ones,
        P_ankle_y,
        P_opp_ankle_y,
        dP_ankle_y,
        ddP_ankle_y,
        q_ankle_z,
        dq_ankle_z,
        ddq_ankle_z
    ], dim=-1)
    return Psi

@dataclass
class GRFMOut:
    F: torch.Tensor         # (B,T,2,3) forces per foot
    M: torch.Tensor         # (B,T,2,3) moments per foot
    mu: torch.Tensor        # (B,T,2)
    cop_local: torch.Tensor # (B,T,2,3)

def grfm_from_kinematics(
    Psi: torch.Tensor,          # (B,T,2,8)
    delta: torch.Tensor,        # (B,T,2,7)
    contact: torch.Tensor,      # (B,T,2) {0,1}
    bodyweight: torch.Tensor,   # (B,)
    foot_geom: torch.Tensor,    # (B,2,3) ll,lw,lh
    R0_ankle: torch.Tensor,     # (B,T,2,3,3)
):
    """
    Appendix-style GRFM with linear coefficients (hardcoded).
    """
    device = Psi.device
    dtype = Psi.dtype
    eta_FY = torch.tensor([0.3116, 3.1785, -2.2963, 0.4151, 0.0088, 0.3374, -0.1206, -0.0089],
                          device=device, dtype=dtype)
    eta_zx = torch.tensor([0.68996, -3.1508, 0.5925, 0.21997, 0.0035, 0.18502, -0.03311, -0.00212],
                          device=device, dtype=dtype)

    FYW_init = (Psi * eta_FY).sum(dim=-1)
    zxl_init = (Psi * eta_zx).sum(dim=-1)
    mu_init = torch.full_like(FYW_init, 0.8)

    dY, dl, dmu, dX, dZ, dhl, ds = delta.unbind(dim=-1)
    mu = torch.clamp(mu_init + dmu, 0.05, 2.0)

    c = contact.to(dtype)
    bw = bodyweight[:, None, None]
    FY = (FYW_init + dY) * bw * c

    ll = foot_geom[..., 0][:, None, :]
    lw = foot_geom[..., 1][:, None, :]
    lh = foot_geom[..., 2][:, None, :]
    zx = (zxl_init + dl) * ll * c

    FX = dX * mu * FY
    inside = torch.clamp(mu.pow(2) * FY.pow(2) - FX.pow(2), min=0.0)
    FZ = dZ * torch.sqrt(inside)

    zy = -torch.abs(dhl * lh) * c
    zz = ds * (lw * 0.5) * c

    cop_local = torch.stack([zx, zy, zz], dim=-1)
    F_global = torch.stack([FX, FY, FZ], dim=-1)

    F_local = (R0_ankle.transpose(-1, -2) @ F_global[..., None])[..., 0]
    M_local = torch.cross(cop_local, F_local, dim=-1)
    M_global = (R0_ankle @ M_local[..., None])[..., 0]
    return GRFMOut(F=F_global, M=M_global, mu=mu, cop_local=cop_local)