import torch
import joblib
import mdtraj as md
import pyemma.coordinates as coor


def kabsch(P, Q):
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
    Vt[d < 0.0, -1] *= -1.0

    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))
    return R, t



def kabsch_rmsd(
    P: torch.Tensor,
    Q: torch.Tensor
):
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B
    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    rmsd = (P_aligned - Q).square().sum(-1).mean(-1).sqrt()
    
    return rmsd