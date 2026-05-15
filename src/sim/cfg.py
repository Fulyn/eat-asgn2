import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.robot.cfg import RobotCfg


@dataclass
class MjRenderConfig:
    height: int = 480
    """camera image height"""
    width: int = 640
    """camera image width"""
    extrinsics: np.ndarray = field(default_factory=lambda: np.eye(4))
    """the extrinsics of the camera"""
    fovy: float = None
    """fov of the y-axis in degrees"""
    save_dir: str = None
    """where to save the rendered images"""

    @staticmethod
    def from_intrinsics_extrinsics(
        height: int,
        width: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> "MjRenderConfig":
        """Note that this will lose some information of intrinsics and extrinsics, but our camera in mujoco doesn't suffer from this"""
        assert intrinsics.shape == (3, 3), "Invalid intrinsics shape"
        assert extrinsics.shape == (4, 4), "Invalid extrinsics shape"

        fovy = (
            2 * np.arctan(0.5 * height / intrinsics[1, 1]) * (180 / np.pi)
        )  # in degrees

        return MjRenderConfig(
            height=height,
            width=width,
            extrinsics=extrinsics,
            fovy=fovy,
        )


@dataclass
class MjSimConfig:
    robot_cfg: RobotCfg
    headless: int = False
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    realtime_sync: bool = False
    use_ground_plane: bool = True
    use_debug_robot: bool = False
    viewer_cfg: Optional[MjRenderConfig] = None

    renderer_cfg: MjRenderConfig = None
