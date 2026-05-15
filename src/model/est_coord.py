from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.local_mlp = nn.Sequential(
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
        self.coord_head = nn.Sequential(
            nn.Conv1d(256 + 256 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )

    def predict_coord(self, pc: torch.Tensor) -> torch.Tensor:
        center = pc.mean(dim=1, keepdim=True)
        x = pc - center
        local = self.local_mlp(x.transpose(1, 2))
        global_feat = local.max(dim=2, keepdim=True).values.expand(-1, -1, pc.shape[1])
        feat = torch.cat([local, global_feat, x.transpose(1, 2)], dim=1)
        return self.coord_head(feat).transpose(1, 2)

    @staticmethod
    def fit_pose(obj_coord: torch.Tensor, cam_pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src = obj_coord - obj_coord.mean(dim=1, keepdim=True)
        dst = cam_pc - cam_pc.mean(dim=1, keepdim=True)
        cov = torch.bmm(src.transpose(1, 2), dst)
        u, _, vh = torch.linalg.svd(cov)
        a = torch.bmm(u, vh)
        det = torch.det(a)
        fix = torch.eye(3, device=cam_pc.device, dtype=cam_pc.dtype).unsqueeze(0).repeat(cam_pc.shape[0], 1, 1)
        fix[:, 2, 2] = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        a = torch.bmm(torch.bmm(u, fix), vh)
        rot = a.transpose(1, 2)
        trans = cam_pc.mean(dim=1) - torch.bmm(obj_coord.mean(dim=1).unsqueeze(1), a).squeeze(1)
        return trans, rot

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pred_coord = self.predict_coord(pc)
        loss = F.smooth_l1_loss(pred_coord, coord)
        mean_abs = (pred_coord - coord).abs().mean()
        metric = dict(
            loss=loss,
            coord_l1=mean_abs.detach(),
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        pred_coord = self.predict_coord(pc)
        trans, rot = self.fit_pose(pred_coord, pc)
        return trans, rot
