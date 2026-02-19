import torch
from dataclasses import dataclass
from typing import List, Tuple
from ..utils.rot import euler_ZXY_to_matrix

@dataclass
class Skeleton47:
    """
    18 segments, pelvis as root.
    DoF layout:
      root: 6  (tx,ty,tz, rz,rx,ry)  -> root rot is ZXY too
      scapula L/R: 2 each
      shoulder L/R: 3 each
      elbow L/R: 2 each
      wrist L/R: 2 each
      hip L/R: 3 each
      knee L/R: 1 each
      ankle L/R: 3 each
      lumbar: 3, thoracic: 3, neck: 3
    Total = 47
    """
    names: List[str]
    parent: List[int]
    dof: List[int]              # per joint/segment rotational dof (root handled separately)
    r_pj: torch.Tensor          # (18,3) parent->joint (in parent frame)
    r_jc: torch.Tensor          # (18,3) joint->CoM (in local frame)

def default_skeleton47(device="cpu", dtype=torch.float32) -> Skeleton47:
    # Order (18):
    names = [
        "pelvis", "lumbar", "thoracic", "neck",
        "scapula_l", "shoulder_l", "elbow_l", "wrist_l",
        "scapula_r", "shoulder_r", "elbow_r", "wrist_r",
        "hip_l", "knee_l", "ankle_l",
        "hip_r", "knee_r", "ankle_r",
    ]
    parent = [
        -1, 0, 1, 2,
        2, 4, 5, 6,
        2, 8, 9, 10,
        0, 12, 13,
        0, 15, 16,
    ]
    # rotational dof per segment joint (excluding root which is 3 rot + 3 trans)
    dof = [
        3, 3, 3, 3,
        2, 3, 2, 2,
        2, 3, 2, 2,
        3, 1, 3,
        3, 1, 3,
    ]
    # Geometry placeholders (meters). Replace with your template.
    r_pj = torch.zeros(18, 3, device=device, dtype=dtype)
    r_jc = torch.zeros(18, 3, device=device, dtype=dtype)

    # Rough kinematic offsets (very approximate, but works)
    # spine chain
    r_pj[1] = torch.tensor([0.0, 0.12, 0.0], device=device, dtype=dtype)
    r_pj[2] = torch.tensor([0.0, 0.12, 0.0], device=device, dtype=dtype)
    r_pj[3] = torch.tensor([0.0, 0.12, 0.0], device=device, dtype=dtype)
    # shoulders from thoracic
    r_pj[4] = torch.tensor([0.10, 0.10, 0.0], device=device, dtype=dtype)
    r_pj[8] = torch.tensor([-0.10, 0.10, 0.0], device=device, dtype=dtype)
    # arms
    r_pj[5]  = torch.tensor([0.10, 0.00, 0.0], device=device, dtype=dtype)
    r_pj[6]  = torch.tensor([0.25, 0.00, 0.0], device=device, dtype=dtype)
    r_pj[7]  = torch.tensor([0.23, 0.00, 0.0], device=device, dtype=dtype)
    r_pj[9]  = torch.tensor([-0.10, 0.00, 0.0], device=device, dtype=dtype)
    r_pj[10] = torch.tensor([-0.25, 0.00, 0.0], device=device, dtype=dtype)
    r_pj[11] = torch.tensor([-0.23, 0.00, 0.0], device=device, dtype=dtype)
    # hips from pelvis
    r_pj[12] = torch.tensor([0.08, -0.08, 0.0], device=device, dtype=dtype)
    r_pj[15] = torch.tensor([-0.08, -0.08, 0.0], device=device, dtype=dtype)
    # legs
    r_pj[13] = torch.tensor([0.0, -0.42, 0.0], device=device, dtype=dtype)
    r_pj[14] = torch.tensor([0.0, -0.42, 0.02], device=device, dtype=dtype)
    r_pj[16] = torch.tensor([0.0, -0.42, 0.0], device=device, dtype=dtype)
    r_pj[17] = torch.tensor([0.0, -0.42, 0.02], device=device, dtype=dtype)

    # CoM offsets (tiny)
    r_jc[:, 1] = 0.02

    return Skeleton47(names=names, parent=parent, dof=dof, r_pj=r_pj, r_jc=r_jc)

def unpack_q_to_local_angles(q: torch.Tensor, skel: Skeleton47) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q: (..., 47) -> root_pos (...,3), local zxy angles for each segment (...,18,3)
    For 2-DoF joints: use [z, x] and y=0
    For 1-DoF joints: use [z] and x=y=0
    """
    root_pos = q[..., 0:3]
    root_rot = q[..., 3:6]  # zxy

    angles = torch.zeros(*q.shape[:-1], 18, 3, device=q.device, dtype=q.dtype)
    angles[..., 0, :] = root_rot

    idx = 6
    for k in range(1, 18):
        d = skel.dof[k]
        if d == 3:
            angles[..., k, :] = q[..., idx:idx+3]
            idx += 3
        elif d == 2:
            angles[..., k, 0:2] = q[..., idx:idx+2]   # z, x
            angles[..., k, 2] = 0.0
            idx += 2
        elif d == 1:
            angles[..., k, 0] = q[..., idx]
            angles[..., k, 1:] = 0.0
            idx += 1
        else:
            raise ValueError("Unsupported DoF")
    assert idx == 47
    return root_pos, angles

def forward_kinematics(q: torch.Tensor, skel: Skeleton47):
    """
    Returns:
      R0k: (...,18,3,3)
      p_joint: (...,18,3)
      p_com: (...,18,3)
    """
    root_pos, angles = unpack_q_to_local_angles(q, skel)
    R0k = torch.zeros(*q.shape[:-1], 18, 3, 3, device=q.device, dtype=q.dtype)
    p_joint = torch.zeros(*q.shape[:-1], 18, 3, device=q.device, dtype=q.dtype)
    p_com = torch.zeros_like(p_joint)

    # root
    R0k[..., 0, :, :] = euler_ZXY_to_matrix(angles[..., 0, :])
    p_joint[..., 0, :] = root_pos
    p_com[..., 0, :] = root_pos + (R0k[..., 0] @ skel.r_jc[0].view(3,1)).squeeze(-1)

    for k in range(1, 18):
        p = skel.parent[k]
        R_local = euler_ZXY_to_matrix(angles[..., k, :])
        R0k[..., k] = R0k[..., p] @ R_local

        offset = (R0k[..., p] @ skel.r_pj[k].view(3,1)).squeeze(-1)
        p_joint[..., k] = p_joint[..., p] + offset

        com_off = (R0k[..., k] @ skel.r_jc[k].view(3,1)).squeeze(-1)
        p_com[..., k] = p_joint[..., k] + com_off

    return R0k, p_joint, p_com