# musclepose/dynamics/euler_jacobians.py
import torch

def jomega_zxy(theta: torch.Tensor) -> torch.Tensor:
    """
    ZXY intrinsic Euler Jacobian mapping theta_dot -> omega (expressed in parent frame).

    theta: (..., 3) = [alpha, beta, gamma]
    returns J: (..., 3, 3)
    """
    a = theta[..., 0]
    b = theta[..., 1]

    ca = torch.cos(a)
    sa = torch.sin(a)
    cb = torch.cos(b)
    sb = torch.sin(b)

    # Jω =
    # [[0,  cos a, -sin a cos b],
    #  [0,  sin a,  cos a cos b],
    #  [1,     0,        sin b]]
    J = torch.zeros((*theta.shape[:-1], 3, 3), dtype=theta.dtype, device=theta.device)
    J[..., 0, 1] = ca
    J[..., 0, 2] = -sa * cb
    J[..., 1, 1] = sa
    J[..., 1, 2] = ca * cb
    J[..., 2, 0] = 1.0
    J[..., 2, 2] = sb
    return J

def djomega_zxy(theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
    """
    Time derivative of ZXY Euler Jacobian: d/dt Jω(theta).

    theta: (...,3), theta_dot: (...,3)
    returns dJ: (...,3,3)
    """
    a = theta[..., 0]
    b = theta[..., 1]
    a_dot = theta_dot[..., 0]
    b_dot = theta_dot[..., 1]

    ca = torch.cos(a)
    sa = torch.sin(a)
    cb = torch.cos(b)
    sb = torch.sin(b)

    dJ = torch.zeros((*theta.shape[:-1], 3, 3), dtype=theta.dtype, device=theta.device)

    # col2 = [cos a, sin a, 0]^T
    # d/dt col2 = a_dot * [-sin a, cos a, 0]
    dJ[..., 0, 1] = -sa * a_dot
    dJ[..., 1, 1] =  ca * a_dot

    # col3 = [-sin a cos b, cos a cos b, sin b]^T
    # d/dt col3 =
    # [-(cos a cos b)*a_dot + (sin a sin b)*b_dot,
    #  -(sin a cos b)*a_dot - (cos a sin b)*b_dot,
    #   (cos b)*b_dot]
    dJ[..., 0, 2] = -(ca * cb) * a_dot + (sa * sb) * b_dot
    dJ[..., 1, 2] = -(sa * cb) * a_dot - (ca * sb) * b_dot
    dJ[..., 2, 2] =  cb * b_dot

    return dJ