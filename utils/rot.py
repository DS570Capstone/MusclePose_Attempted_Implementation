import torch

def skew(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy, vx,  O], dim=-1),
    ], dim=-2)

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)

def rot_x(a):
    ca, sa = torch.cos(a), torch.sin(a)
    O = torch.zeros_like(a); I = torch.ones_like(a)
    return torch.stack([
        torch.stack([I, O, O], dim=-1),
        torch.stack([O, ca, -sa], dim=-1),
        torch.stack([O, sa,  ca], dim=-1),
    ], dim=-2)

def rot_y(a):
    ca, sa = torch.cos(a), torch.sin(a)
    O = torch.zeros_like(a); I = torch.ones_like(a)
    return torch.stack([
        torch.stack([ ca, O, sa], dim=-1),
        torch.stack([ O,  I, O ], dim=-1),
        torch.stack([-sa, O, ca], dim=-1),
    ], dim=-2)

def rot_z(a):
    ca, sa = torch.cos(a), torch.sin(a)
    O = torch.zeros_like(a); I = torch.ones_like(a)
    return torch.stack([
        torch.stack([ca, -sa, O], dim=-1),
        torch.stack([sa,  ca, O], dim=-1),
        torch.stack([O,   O,  I], dim=-1),
    ], dim=-2)

def euler_ZXY_to_matrix(zxy: torch.Tensor) -> torch.Tensor:
    """
    ISB-ish: z = flex/ext, x = abd/add, y = int/ext rotation
    R = Rz(z) Rx(x) Ry(y)
    zxy: (...,3)
    """
    z, x, y = zxy[..., 0], zxy[..., 1], zxy[..., 2]
    return rot_z(z) @ rot_x(x) @ rot_y(y)

def safe_norm(x, eps=1e-8):
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))