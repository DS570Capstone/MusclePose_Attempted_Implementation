import torch
from dataclasses import dataclass
from torch.func import jacrev
from MusclePose.utils.rot import skew
from MusclePose.physics.euler_jacobian import jomega_zxy, djomega_zxy
from MusclePose.physics.skeleton47 import forward_kinematics, default_skeleton47, unpack_q_to_local_angles

@dataclass
class IDOut:
    tau_q: torch.Tensor   # (B,T,47)
    M: torch.Tensor       # (B,T,47,47)
    C: torch.Tensor       # (B,T,47)
    Fext: torch.Tensor    # (B,T,47)

def _spatial_inertia(mk, I0k, R0k):
    """
    mk: (B,18)
    I0k:(B,18,3,3)
    R0k:(B,18,3,3)
    returns Mk: (B,18,6,6)
    """
    B,K = mk.shape
    I3 = torch.eye(3, device=mk.device, dtype=mk.dtype)[None,None]
    Iw = R0k @ I0k @ R0k.transpose(-1,-2)
    Mk = torch.zeros((B,K,6,6), device=mk.device, dtype=mk.dtype)
    Mk[:,:,0:3,0:3] = mk[:,:,None,None] * I3
    Mk[:,:,3:6,3:6] = Iw
    return Mk

def _com_positions_from_q(q, skel):
    # q: (47,)
    R0k, p_joint, p_com = forward_kinematics(q[None, :], skel)
    return p_com[0]  # (18,3)

def _angles_from_q(q, skel):
    # return segment rotations (18,3,3)
    R0k, _, _ = forward_kinematics(q[None,:], skel)
    return R0k[0]


def _joint_rot_column_spans(skel):
    spans = [(3, 6)]  # root rotation columns in q
    idx = 6
    for k in range(1, len(skel.dof)):
        d = skel.dof[k]
        spans.append((idx, idx + d))
        idx += d
    return spans


def _angular_jacobians(theta_local, theta_dot_local, R0k, skel, spans, total_dof):
    """
    Build angular-velocity Jacobians by accumulating through the kinematic chain.
    Each segment inherits the angular Jacobian of its parent plus its own joint contribution.

    theta_local: (18, 3)  local Euler angles per segment
    theta_dot_local: (18, 3)  local Euler angle rates per segment
    R0k: (18, 3, 3)  world-frame orientations per segment
    skel: Skeleton47
    spans: list[(start, stop)]  column ranges in q for each segment's rotational DoFs
    total_dof: 47
    """
    K = len(spans)
    device = theta_local.device
    dtype = theta_local.dtype
    JO = torch.zeros((K, 3, total_dof), device=device, dtype=dtype)
    dJO = torch.zeros_like(JO)

    for k in range(K):
        start, stop = spans[k]
        width = stop - start
        if width <= 0:
            continue

        # Parent world rotation (identity for root)
        p = skel.parent[k]
        if p < 0:
            R_parent = torch.eye(3, device=device, dtype=dtype)
        else:
            R_parent = R0k[p]  # (3, 3)

        # Local Euler Jacobian and its time derivative
        Jw = jomega_zxy(theta_local[k])                        # (3, 3)
        dJw = djomega_zxy(theta_local[k], theta_dot_local[k])  # (3, 3)

        # This joint's contribution rotated into world frame
        JO[k, :, start:stop] = R_parent @ Jw[:, :width]
        dJO[k, :, start:stop] = R_parent @ dJw[:, :width]

        # Inherit all ancestor contributions from parent
        if p >= 0:
            JO[k] = JO[k] + JO[p]
            dJO[k] = dJO[k] + dJO[p]

    return JO, dJO

def inverse_dynamics_autograd(
    q: torch.Tensor, qdot: torch.Tensor, qddot: torch.Tensor,
    mk: torch.Tensor, I0k: torch.Tensor,
    foot_wrenches: torch.Tensor = None,   # (B,T,2,6) per foot [Fx,Fy,Fz,Mx,My,Mz] in world
    foot_indices=(14,17),                 # ankle_l, ankle_r segments in our skeleton list
):
    """
    q,qdot,qddot: (B,T,47)
    mk:  (B,18)
    I0k: (B,18,3,3)
    """
    skel = default_skeleton47(device=q.device, dtype=q.dtype)
    B,T,N = q.shape
    K = 18
    spans = _joint_rot_column_spans(skel)
    _, all_angles = unpack_q_to_local_angles(q, skel)
    _, all_angle_rates = unpack_q_to_local_angles(qdot, skel)

    # jacobians via jacrev: JV_k = d p_com_k / d q
    # We'll compute for each frame independently.
    tau_out = torch.zeros((B,T,N), device=q.device, dtype=q.dtype)
    M_out = torch.zeros((B,T,N,N), device=q.device, dtype=q.dtype)
    C_out = torch.zeros((B,T,N), device=q.device, dtype=q.dtype)
    Fext_out = torch.zeros((B,T,N), device=q.device, dtype=q.dtype)

    # define jacobian fns (per-sample)
    Jpos_fn = jacrev(lambda qq: _com_positions_from_q(qq, skel))   # (18,3) wrt (47)
    for b in range(B):
        for t in range(T):
            qq = q[b, t]
            Jpos = Jpos_fn(qq)  # (18,3,47)
            JV = Jpos           # (18,3,47)

            # FK for R0k — needed for both inertia rotation and angular Jacobians
            R0k, _, _ = forward_kinematics(qq[None, :], skel)
            R0k = R0k[0]  # (18,3,3)

            theta_local = all_angles[b, t]
            theta_dot_local = all_angle_rates[b, t]
            JO, dJO = _angular_jacobians(theta_local, theta_dot_local, R0k, skel, spans, N)
            if 0 < t < T-1:
                Jpos_prev = Jpos_fn(q[b, t-1])
                Jpos_next = Jpos_fn(q[b, t+1])
                dJV = (Jpos_next - Jpos_prev) / 2.0
            elif t == 0:
                dJV = (Jpos_fn(q[b, t+1]) - Jpos) 
            else:
                dJV = (Jpos - Jpos_fn(q[b, t-1]))

            Mk_sp = _spatial_inertia(mk[b:b+1], I0k[b:b+1], R0k[None, ...])[0]  # (18,6,6)

            J = torch.cat([JV, JO], dim=1)      # (18,6,47)
            dJ = torch.cat([dJV, dJO], dim=1)

            JT = J.transpose(-1, -2)            # (18,47,6)
            Mseg = JT @ Mk_sp @ J               # (18,47,47)
            Mmat = Mseg.sum(dim=0)              # (47,47)

            # Coriolis term (approx, since JO=0 => Omega=0 => block term vanishes)
            term1 = (JT @ Mk_sp @ dJ)           # (18,47,47)
            Cvec = (term1 @ qdot[b,t][:,None]).squeeze(-1).sum(dim=0)  # (47,)

            # External forces in generalized coords (feet only)
            Fext = torch.zeros((47,), device=q.device, dtype=q.dtype)
            if foot_wrenches is not None:
                # Build foot Jacobians Jfoot = [JV_foot; JO_foot] using the corresponding segment rows
                # We use the CoM Jacobian as a proxy for the foot point.
                for fi, seg_k in enumerate(foot_indices):
                    Jfoot = torch.cat([JV[seg_k], JO[seg_k]], dim=0)     # (6,47)
                    wrench = foot_wrenches[b,t,fi]                        # (6,)
                    Fext = Fext + (Jfoot.transpose(0,1) @ wrench[:,None]).squeeze(-1)

            # Gravity generalized force: Q_grav = sum_k mk * JV_k^T @ g
            g_vec = torch.tensor([0.0, -9.81, 0.0], device=q.device, dtype=q.dtype)
            Grav = torch.zeros((N,), device=q.device, dtype=q.dtype)
            for seg_k in range(K):
                Grav = Grav + mk[b, seg_k] * (JV[seg_k].T @ g_vec)

            # tau = M*qddot + C - Q_grav - Q_ext
            tau = (Mmat @ qddot[b,t][:,None]).squeeze(-1) + Cvec - Grav - Fext

            tau_out[b,t] = tau
            M_out[b,t] = Mmat
            C_out[b,t] = Cvec
            Fext_out[b,t] = Fext

    return IDOut(tau_q=tau_out, M=M_out, C=C_out, Fext=Fext_out)