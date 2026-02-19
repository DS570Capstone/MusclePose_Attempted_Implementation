import torch
def make_tokens_from_coco(kxy: torch.Tensor, kconf: torch.Tensor):
    """
    kxy:   (B,T,17,2) pixel coords
    kconf: (B,T,17)
    Returns tokens: (B,T,D)
    """
    B,T,_,_ = kxy.shape
    pelvis = 0.5 * (kxy[:,:,11] + kxy[:,:,12])                    # (B,T,2)
    spread = (kxy - pelvis[:,:,None,:]).norm(dim=-1)              # (B,T,17)
    s = spread.median(dim=-1).values.clamp(min=1e-6)              # (B,T)

    kxy_n = (kxy - pelvis[:,:,None,:]) / s[:,:,None,None]
    feat = torch.cat([kxy_n.reshape(B,T,-1), kconf.reshape(B,T,-1)], dim=-1)

    d1 = torch.zeros_like(kxy_n)
    d2 = torch.zeros_like(kxy_n)
    d1[:,1:] = kxy_n[:,1:] - kxy_n[:,:-1]
    d2[:,2:] = kxy_n[:,2:] - 2*kxy_n[:,1:-1] + kxy_n[:,:-2]

    feat = torch.cat([feat, d1.reshape(B,T,-1), d2.reshape(B,T,-1)], dim=-1)
    return feat