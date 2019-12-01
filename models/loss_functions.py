import torch


def dist_loss(points):
    P = points
    Pb = P.roll(1, dims=2)

    D = (P - Pb) ** 2

    return torch.sum(D, dim=[-2, -1]).mean()


def curvature_loss(points):
    P = points
    Pf = P.roll(-1, dims=2)
    Pb = P.roll(1, dims=2)

    K = Pf + Pb - 2 * P

    return torch.norm(K, dim=-1).mean()
