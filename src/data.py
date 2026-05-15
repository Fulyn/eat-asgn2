import os
from typing import Dict, Optional
import random
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .constants import DEPTH_IMG_SCALE, TABLE_HEIGHT, PC_MAX, PC_MIN, OBJ_INIT_TRANS
from .utils import get_pc, get_workspace_mask
from .vis import Vis
from .robot.cfg import get_robot_cfg


class PoseDataset(Dataset):
    def __init__(self, config: Config, mode: str, scale: int = 1):
        """
        Dataset for pose estimation

        Parameters
        ----------
        config: Config
            Configuration object
        mode: str
            Mode of the dataset (e.g. train or val)
        scale: int
            Scale of the dataset, used to make the dataset larger
            so that we don't need to wait for the restart of the dataloader
        """
        super().__init__()
        self.config = config
        self.robot_cfg = get_robot_cfg(config.robot)
        self.data_root = os.path.join("data", config.obj_name, mode)
        self.files = sorted(os.listdir(self.data_root))

        self.data = dict()
        for f in tqdm(self.files, desc="Loading data"):
            try:
                fdir = os.path.join(self.data_root, f)

                obj_pose = np.load(os.path.join(fdir, "object_pose.npy"))
                if not np.linalg.norm(obj_pose[:2, 3] - OBJ_INIT_TRANS[:2]) < 0.1:
                    # some times the object will be out of the workspace
                    # so we need to skip this sample
                    # this rarely happens so we don't need to worry about it
                    continue
                camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
                depth_array = (
                    np.array(
                        cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
                    )
                    / DEPTH_IMG_SCALE
                )

                full_pc_camera = get_pc(
                    depth_array, self.robot_cfg.camera_cfg.intrinsics
                ) * np.array([-1, -1, 1])
                full_pc_world = (
                    np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
                    + camera_pose[:3, 3]
                )
                full_coord = np.einsum(
                    "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
                )

                pc_mask = get_workspace_mask(full_pc_world)

                self.data[f] = (
                    full_pc_camera[pc_mask],
                    full_coord[pc_mask],
                    camera_pose,
                    obj_pose,
                )
            except Exception as e:
                print(f"Error loading file {f}: {e}")

        self.files = self.files * scale
        random.shuffle(self.files)

    def __len__(self) -> int:
        """
        For a torch dataset, a __len__ is required.

        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.files)

    def __getitem__(self, idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        For a torch dataset, a __getitem__ is required.

        Parameters
        ----------
        idx: Optional[int]
            Index of the item to get. If None, a random index is used.

        Returns
        -------
        A dict of:

            pc: the point cloud in camera frame with shape (N, 3)

            trans: the ground truth translation vector with shape (3,)

            rot: the ground truth rotation matrix with shape (3, 3)

            coord: the ground truth coordinates in the object frame with shape (N, 3)

            camera_pose: the camera pose with shape (4, 4) (Used in simulation evaluation)

            obj_pose_in_world: the object pose in world frame with shape (4, 4) (Used in simulation evaluation)

        Note that they will be converted to torch tensors in the dataloader.

        The shape will be (B, ...) for batch size B when you get the data from the dataloader.
        """
        if idx is None:
            idx = random.randint(0, len(self.files) - 1)
        if self.files[idx] not in self.data:
            return self.__getitem__()

        mask_pc_camera, mask_coord, camera_pose, obj_pose = self.data[self.files[idx]]

        sel_pc_idx = np.random.randint(0, len(mask_pc_camera), self.config.point_num)

        pc_camera = mask_pc_camera[sel_pc_idx]
        coord = mask_coord[sel_pc_idx]
        rel_obj_pose = np.linalg.inv(camera_pose) @ obj_pose

        return dict(
            pc=pc_camera.astype(np.float32),
            coord=coord.astype(np.float32),
            trans=rel_obj_pose[:3, 3].astype(np.float32),
            rot=rel_obj_pose[:3, :3].astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            obj_pose_in_world=obj_pose.astype(np.float32),
        )


class Loader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get(self) -> dict:
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data
