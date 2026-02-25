import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from MusclePose.physics.anthropometrics import anthropometrics_from_beta_E
from MusclePose.physics.inverse_dynamics_autograd import inverse_dynamics_autograd
from MusclePose.physics.grfm import grfm_from_kinematics
from MusclePose.physics.mtg import mtg_torque

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
    phi: torch.Tensor = None


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) with positional encoding added."""
        return x + self.pe[:, :x.size(1)]


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: queries attend to key/value context.

    query  (B, N, D)  -- learned event tokens
    context(B, T, D)  -- fused physics sequence
    output (B, N, D)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # pre-norm cross-attention
        q = self.norm_q(query)
        ctx = self.norm_ctx(context)
        attn_out, _ = self.cross_attn(q, ctx, ctx)
        query = query + attn_out
        # feed-forward with residual
        query = query + self.ff(self.norm_ff(query))
        return query


class PhysicsBridge(nn.Module):
    """Learned Cross-Attention Bridge with Temporal Event Tokens.

    Instead of naively average-pooling the fused physics sequence,
    this module uses *learned temporal event tokens* as cross-attention
    queries that attend to the full time-resolved physics context.
    Each event token can specialise on a different temporal aspect
    (rep phases, transitions, peak-force moments, etc.), producing
    richer soft tokens for the downstream LLM.

    Architecture:
      1. physics_proj:  project raw physics vector (266-d) -> d_model
      2. fuse:          concatenate encoder hidden states + physics proj,
                        then project back to d_model
      3. temporal_pe:   sinusoidal positional encoding added to fused seq
      4. event_tokens:  learned queries  (n_tokens, d_model)
      5. cross_attn:    N layers of multi-head cross-attention
                        query = event_tokens,  key/value = fused sequence
      6. to_llm:        project d_model -> llm_dim

    Inputs consumed (from ForwardOut):
      phi      (B, T, 256)   encoder hidden states
      q        (B, T, 47)    joint angles
      qdot     (B, T, 47)    joint velocities
      tau_q    (B, T, 47)    inverse-dynamics torques
      tau_mtg  (B, T, 41)    muscle-model torques
      alpha    (B, T, 82)    muscle activations (flex + ext)
      contact  (B, T, 2)     foot contact

    Output:
      soft_tokens (B, n_tokens, llm_dim)  ready to prepend to LLM input
    """

    PHYSICS_DIM = 47 + 47 + 47 + 41 + 82 + 2  # 266

    def __init__(
        self,
        d_model: int = 256,
        llm_dim: int = 3072,
        n_tokens: int = 8,
        n_cross_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        # ---- physics projection ----
        self.physics_proj = nn.Sequential(
            nn.Linear(self.PHYSICS_DIM, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # ---- encoder + physics fusion ----
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ---- temporal positional encoding ----
        self.temporal_pe = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # ---- learned temporal event tokens (queries) ----
        self.event_tokens = nn.Parameter(
            torch.randn(1, n_tokens, d_model) * 0.02
        )

        # ---- cross-attention layers ----
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_cross_layers)
        ])

        # ---- final layer-norm before LLM projection ----
        self.out_norm = nn.LayerNorm(d_model)

        # ---- project to LLM embedding space ----
        self.to_llm = nn.Sequential(
            nn.Linear(d_model, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # ---- output normalisation: match LLM embedding scale ----
        # LLM text embeddings typically have std ≈ 0.03.  The learned gate
        # starts at 0.0 (Tanh → 0) so that at init the soft tokens are silent,
        # then gradually opens during training.
        self.out_ln = nn.LayerNorm(llm_dim)
        self.out_gate = nn.Parameter(torch.tensor(0.5))   # scalar gate, init=0.5 for nonzero soft tokens at start

    def forward(self, out: 'ForwardOut') -> torch.Tensor:
        # ---- build physics vector (B, T, 266) ----
        physics_vec = torch.cat([
            out.q,
            out.qdot,
            out.tau_q,
            out.tau_mtg,
            out.alpha,
            out.contact,
        ], dim=-1)

        # ---- project & fuse ----
        p = self.physics_proj(physics_vec)                       # (B, T, d_model)
        fused = self.fuse(torch.cat([out.phi, p], dim=-1))       # (B, T, d_model)

        # ---- add temporal positional encoding ----
        context = self.temporal_pe(fused)                        # (B, T, d_model)

        # ---- expand learned event tokens per batch ----
        B = context.size(0)
        queries = self.event_tokens.expand(B, -1, -1)           # (B, n_tokens, d_model)

        # ---- cross-attention: event tokens attend to context ----
        for layer in self.cross_layers:
            queries = layer(queries, context)                    # (B, n_tokens, d_model)

        # ---- project to LLM space with scale gating ----
        raw = self.to_llm(self.out_norm(queries))                # (B, n_tokens, llm_dim)
        normed = self.out_ln(raw)                                # layer-norm to ~unit var
        gate = torch.tanh(self.out_gate)                         # in [-1, 1], starts at 0
        return normed * gate * 0.03                              # scale to match LLM embed std

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
        # In the paper Psi uses ankle vertical kinematics + plantar/dorsi angle.
        # Here we derive a simple proxy from q states.
        Psi = torch.zeros((B,T,2,8), device=tokens.device, dtype=tokens.dtype)
        Psi[...,0] = 1.0
        Psi[...,1] = q[...,1:2].repeat(1,1,2)
        Psi[...,3] = qdot[...,1:2].repeat(1,1,2)
        Psi[...,4] = qddot[...,1:2].repeat(1,1,2)

        # Ankle world orientation placeholder: identity
        R0_ankle = torch.eye(3, device=tokens.device, dtype=tokens.dtype)[None,None,None,:,:].repeat(B,T,2,1,1)

        grfm = grfm_from_kinematics(Psi, delta, contact, anth.W, foot_geom, R0_ankle)
        # pack foot wrenches for ID: (B,T,2,6)
        foot_wrenches = torch.cat([grfm.F, grfm.M], dim=-1)

        # Inverse dynamics τ_q  —  tau_q is detached in the loss function,
        # so no gradients flow back through this (jacrev-based) computation.
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
            tau_q=idout.tau_q, tau_mtg=tau_mtg, phi=phi
        )