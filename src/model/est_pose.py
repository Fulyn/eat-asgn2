from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotation representation to rotation matrices."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1,
        dim=-1,
        eps=1e-6,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = torch.bmm(pred.transpose(1, 2), target)
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos).mean()


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(259, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),
        )

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pred_trans, pred_rot = self.est(pc)
        trans_loss = F.smooth_l1_loss(pred_trans, trans)
        rot_loss = F.mse_loss(pred_rot, rot)
        loss = 20.0 * trans_loss + 5.0 * rot_loss
        metric = dict(
            loss=loss,
            trans_loss=trans_loss.detach(),
            rot_loss=rot_loss.detach(),
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        center = pc.mean(dim=1)
        x = (pc - center[:, None, :]).transpose(1, 2)
        feat = self.point_mlp(x).max(dim=2).values
        out = self.head(torch.cat([feat, center], dim=1))
        trans = center + out[:, :3]
        rot = rot6d_to_matrix(out[:, 3:9])
        return trans, rot
