import torch

def tau_omega(qdot, wmax, g1, g2, g3):
    ecc = ((1 - g1) * torch.abs(wmax) - (g2 + 1) * g1 * g3 * qdot) / (
           (1 - g1) * torch.abs(wmax) + (g2 + 1) * g1 * qdot + 1e-8
    )
    con = (torch.abs(wmax) - qdot) / (torch.abs(wmax) + g2 * qdot + 1e-8)
    return torch.where(qdot < 0, ecc, con)

def tau_theta(q, g4, g5, g6):
    poly = g4 + g5*q + g6*(q**2)
    return torch.clamp(poly, min=0.0)

def tau_passive(q, qmin, qmax, qdot, g10, g11, g12, g13, g14):
    return (g10 * torch.exp(-g11*(q - qmin))
            - g12 * torch.exp( g13*(q - qmax))
            - g14 * qdot)

def mtg_torque(q, qdot, alpha_flex, alpha_ext, tau0, wmax, qmin, qmax, gamma_active, gamma_passive):
    """
    gamma_active: (...,6)  [g1 g2 g3 g4 g5 g6]
    gamma_passive:(...,5)  [g10 g11 g12 g13 g14]
    torque sign: flex positive, ext negative
    """
    g1,g2,g3,g4,g5,g6 = [gamma_active[...,i] for i in range(6)]
    g10,g11,g12,g13,g14 = [gamma_passive[...,i] for i in range(5)]

    tw = tau_omega(qdot, wmax, g1, g2, g3)
    tt = tau_theta(q, g4, g5, g6)
    t_active = (alpha_flex - alpha_ext) * tw * tt * tau0
    t_pass = tau_passive(q, qmin, qmax, qdot, g10, g11, g12, g13, g14)
    return t_active + t_pass