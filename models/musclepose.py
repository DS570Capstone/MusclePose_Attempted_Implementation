import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from ..physics.anthropometrics import anthropometrics_from_beta_E
from ..physics.inverse_dynamics_autograd import inverse_dynamics_autograd
from ..physics.grfm import grfm_from_kinematics
from ..physics.mtg import mtg_torque

@dataclass
class ForwardOut:
    q: torch.Tensor
    qdot: torch.Tensor
    qddot: torch.Tensor
    contact: torch.Tensor
    E: torch.Tensor
    delta: torch.Tensor
    Z: torch.Tensor
    alpha: torch.Tensor
    tau_q: torch.Tensor
    tau_mtg: torch.Tensor

def finite_differences(x: torch.Tensor, dt: float):
    xdot = torch.zeros_like(x)
    xddot = torch.zeros_like(x)
    xdot[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2*dt)
    xdot[:, 0] = (x[:, 1] - x[:, 0]) / dt
    xdot[:, -1] = (x[:, -1] - x[:, -2]) / dt
    xddot[:, 1:-1] = (x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]) / (dt*dt)
    xddot[:, 0] = xddot[:, 1]
    xddot[:, -1] = xddot[:, -2]
    return xdot, xddot

class MotionEncoder(nn.Module):
    def __init__(self, d_in, d_model=256, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(self.proj(x))

class MusclePoseCOCO(nn.Module):
    """
    COCO-2D -> q (47DoF) + physics outputs
    """
    def __init__(self, d_in, dt=1/30,
                 delta_dim=7,       # per foot
                 E_dim=18,          # minimal for mass residuals
                 Z_dim=41,          # rotational dofs count (47-6)
                 alpha_dim=2*41):   # flex/ext per rotational dof
        super().__init__()
        self.dt = dt
        self.enc = MotionEncoder(d_in=d_in, d_model=256, n_layers=8, n_heads=8)

        # heads
        self.head_q = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 47))
        self.head_contact = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))  # L/R
        self.head_delta = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2*delta_dim))
        self.head_alpha = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, alpha_dim))

        # sequence-level heads from pooled phi
        self.head_beta = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 10))
        self.head_E = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, E_dim))
        self.head_Z = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, Z_dim))

        # MTG coefficient placeholders (you should replace with your per-DoF gammas)
        self.register_buffer("gamma_active", torch.tensor([1.2,0.2,0.3, 1.0,0.0,-0.1]).float())  # (6,)
        self.register_buffer("gamma_passive", torch.tensor([0.5,1.0,0.4,1.0,0.05]).float())      # (5,)

        # joint limits placeholders for 41 rotational DoFs
        self.register_buffer("qmin", -torch.ones(41))
        self.register_buffer("qmax",  torch.ones(41))
        self.register_buffer("wmax",  10*torch.ones(41))
        self.register_buffer("tau0_base", 100*torch.ones(41))

    def forward(self, tokens: torch.Tensor):
        """
        tokens: (B,T,D) built from COCO keypoints
        """
        B,T,D = tokens.shape
        phi = self.enc(tokens)                   # (B,T,256)
        phi_pool = phi.mean(dim=1)               # (B,256)

        q = self.head_q(phi)                     # (B,T,47)
        qdot, qddot = finite_differences(q, self.dt)

        contact_logits = self.head_contact(phi)  # (B,T,2)
        contact = (torch.sigmoid(contact_logits) > 0.5).float()

        delta = self.head_delta(phi).view(B,T,2,-1)
        delta = torch.tanh(delta)                # ensure [-1,1]

        alpha = self.head_alpha(phi)             # (B,T,2*41)
        alpha = torch.sigmoid(alpha)

        beta = self.head_beta(phi_pool)          # (B,10)
        E = self.head_E(phi_pool)                # (B,E_dim)
        Z = self.head_Z(phi_pool)                # (B,41)

        # Anthropometrics
        anth = anthropometrics_from_beta_E(beta, E)

        # Foot geometry placeholders (B,2,3): ll,lw,lh
        foot_geom = torch.tensor([0.25,0.10,0.05], device=tokens.device, dtype=tokens.dtype)[None,None,:].repeat(B,2,1)

        # Build Psi for each foot: (B,T,2,8) placeholder features from q
        # In your paper Psi uses ankle vertical kinematics + plantar/dorsi angle.
        # Here we derive a simple proxy from q states.
        Psi = torch.zeros((B,T,2,8), device=tokens.device, dtype=tokens.dtype)
        Psi[...,0] = 1.0
        # crude: use root y, root ydot, root yddot
        Psi[...,1] = q[...,1:2].repeat(1,1,2)
        Psi[...,3] = qdot[...,1:2].repeat(1,1,2)
        Psi[...,4] = qddot[...,1:2].repeat(1,1,2)

        # Ankle world orientation placeholder: identity
        R0_ankle = torch.eye(3, device=tokens.device, dtype=tokens.dtype)[None,None,None,:,:].repeat(B,T,2,1,1)

        grfm = grfm_from_kinematics(Psi, delta, contact, anth.W, foot_geom, R0_ankle)
        # pack foot wrenches for ID: (B,T,2,6)
        foot_wrenches = torch.cat([grfm.F, grfm.M], dim=-1)

        # Inverse dynamics τ_q
        idout = inverse_dynamics_autograd(q, qdot, qddot, anth.mk, anth.I0k, foot_wrenches=foot_wrenches)

        # MTG torques τ_mtg for 41 rotational DoFs (q[6:])
        qrot = q[..., 6:]         # (B,T,41)
        qrot_dot = qdot[..., 6:]
        alpha_f = alpha[..., 0:41]
        alpha_e = alpha[..., 41:82]

        # tau0 with offsets Z (sequence-level)
        tau0 = self.tau0_base[None,None,:] * (1.0 + 0.2*torch.tanh(Z[:,None,:]))

        ga = self.gamma_active[None,None,None,:].repeat(B,T,41,1)
        gp = self.gamma_passive[None,None,None,:].repeat(B,T,41,1)

        tau_mtg = mtg_torque(
            qrot, qrot_dot, alpha_f, alpha_e,
            tau0, self.wmax[None,None,:],
            self.qmin[None,None,:], self.qmax[None,None,:],
            ga, gp
        )  # (B,T,41)

        return ForwardOut(
            q=q, qdot=qdot, qddot=qddot,
            contact=contact, E=E, delta=delta, Z=Z, alpha=alpha,
            tau_q=idout.tau_q, tau_mtg=tau_mtg
        )