import argparse
import numpy as np
import cv2
from tqdm import trange
try:
    from transforms3d.axangles import axangle2mat, mat2axangle
except ModuleNotFoundError:
    def axangle2mat(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        C = 1 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ]
        )

    def mat2axangle(mat):
        angle = np.arccos(np.clip((np.trace(mat) - 1) / 2, -1, 1))
        axis = np.array(
            [
                mat[2, 1] - mat[1, 2],
                mat[0, 2] - mat[2, 0],
                mat[1, 0] - mat[0, 1],
            ]
        )
        norm = np.linalg.norm(axis)
        axis = np.array([1.0, 0.0, 0.0]) if norm < 1e-8 else axis / norm
        return axis, angle

# You might find some of the currently unused functions useful
# You are also free to add additional imports
# If you want to use packages that are not installed by default, please add an ```Install.md``` to show how to install them
from src.type import Grasp
from src.constants import CALIB_GRID_SCALE, CALIB_CHESSBOARD_SIZE
from src.sim.grasp_env import Obs, GraspEnvConfig, GraspEnv, get_grasps
from src.utils import to_pose
from src.vis import Vis

# The visualization functions might be useful for debugging
# It is not mandatory to use them
def draw_projected_points(
    img: np.ndarray,
    object_points: np.ndarray,
    rvec: np.ndarray, 
    tvec: np.ndarray, 
    K: np.ndarray, 
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 4 
) -> np.ndarray:
    """
    Project 3D points in the object frame onto the image plane and draw them
    on the input image.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, 3).
    object_points : np.ndarray
        3D points of shape (K, 3) expressed in the object frame.
    rvec : np.ndarray
        Rotation of the object in the camera frame in axis angle with shape (3, 1) (use opencv convention).
    tvec : np.ndarray
        Translation of the object in the camera frame with shape (3, 1).
    K : np.ndarray
        Camera intrinsic matrix of shape (3, 3).
    color : tuple[int, int, int], optional
        Color used to draw the projected points.
    radius : int, optional
        Radius of the drawn points in pixels.

    Returns
    -------
    np.ndarray
        A copy of the input image with the projected 2D points drawn on it.

    Notes
    -----
    This function is useful for visually checking whether the estimated pose
    between the camera and the chessboard is correct. If the projected points 
    align well with the chessboard's inner corners, the pose estimate is 
    likely reasonable.
    """
    imgpts, _ = cv2.projectPoints(
        object_points.astype(np.float32),
        rvec,
        tvec,
        K,
        None,
    )
    imgpts = imgpts.reshape(-1, 2)

    out = img.copy()
    for p in imgpts:
        x, y = np.round(p).astype(int)
        cv2.circle(out, (x, y), radius, color, -1)
    return out

def visualize_result(result: np.ndarray, gripper_poses: np.ndarray, chessboard_poses: np.ndarray):
    """
    Check whether the pose estimation between the gripper and the chessboard is correct
    With the estimated pose between the gripper and the chessboard, we can calculate the pose of the chessboard
    If the result is correct, then the estimated pose of the chessboard should be constant over different observations

    Parameters
    ----------
    result : np.ndarray
        The estimated pose of the camera in the gripper frame, (4, 4)
    gripper_poses : np.ndarray
        The poses of the gripper in the base frame, (N, 4, 4)
    chessboard_poses : np.ndarray
        The poses of the chessboard in the camera frame, (N, 4, 4)
    """
    vis = Vis()
    vis_lst = []
    marker_poses_in_world = []
    for gripper_pose, marker_pose in zip(gripper_poses, chessboard_poses):
        marker_pose_in_world = gripper_pose @ result @ marker_pose
        marker_poses_in_world.append(marker_pose_in_world)
        vis_lst += vis.pose(marker_pose_in_world[:3, 3], marker_pose_in_world[:3, :3])
    print("std of chessboard pose in world: ")
    print(np.array(marker_poses_in_world).std(axis=0))
    vis.show(vis_lst)

def find_chessboard_corners(
    img: np.ndarray,
    chessboard_size: tuple[int, int],
) -> tuple[bool, np.ndarray]:
    """
    Find the inner corners of the chessboard on the input image

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, 3).
    chessboard_size : tuple[int, int]
        Number of inner corners for each row and column.

    Returns
    -------
    tuple[bool, np.ndarray]
        A tuple containing a boolean indicating whether the chessboard was found and the corners of the chessboard (with shape (K, 1, 2), in pixel).
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    succ, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # We can add sub-pixel refinement (cv2.cornerSubPix) here to improve the accuracy
    # But we found that in simulation the sub-pixel refinement is not necessary
    # So we don't add it for simplicity
    return succ, corners

def get_corners_in_chessboard_frame(
    chessboard_size: tuple[int, int],
    chessboard_grid_scale: float,
) -> np.ndarray:
    """
    Get the corners of the chessboard in the chessboard's local frame

    Parameters
    ----------
    chessboard_size : tuple[int, int]
        Number of inner corners for each row and column.
    chessboard_grid_scale : float
        Scale of the chessboard grid.

    Returns
    -------
    np.ndarray
        The corners of the chessboard in the chessboard frame (with shape (K, 3)).
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= chessboard_grid_scale
    return objp

def solve_pnp_checkboard(
    chessboard_corners: np.ndarray,
    corners: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Solve the PnP problem for a chessboard

    Parameters
    ----------
    chessboard_corners : np.ndarray
        Corners of the chessboard in the chessboard frame, with shape (K, 3).
    corners : np.ndarray
        Corners of the chessboard, with shape (K, 1, 2), in pixel.
    intrinsics : np.ndarray
        Camera intrinsic matrix, with shape (3, 3).

    Returns
    -------
    tuple[bool, np.ndarray, np.ndarray]
        A tuple containing a boolean indicating whether the PnP problem was solved, the rotation vector (with shape (3, 1), in axis angle) and the translation vector (with shape (3, 1)) of the chessboard in the camera frame.
    """
    ret, rvec, tvec = cv2.solvePnP(chessboard_corners, corners, intrinsics, None)
    return ret, rvec, tvec

def calibrate_hand_eye(
    R_gripper: np.ndarray,
    t_gripper: np.ndarray,
    R_chessboard: np.ndarray,
    t_chessboard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hand-eye calibration 
    we use the method introduced by
    R. Y. Tsai and R. K. Lenz. A new technique for fully autonomous and efficient 3d robotics hand/eye calibration. IEEE Transactions on Robotics and Automation, 5(3):345–358, June 1989.

    Parameters
    ----------
    R_gripper : np.ndarray
        Rotation matrices of the gripper in the base frame (with shape (N, 3, 3)).
    t_gripper : np.ndarray
        Translation of the gripper in the base frame (with shape (N, 3, 1)).
    R_chessboard matrices : np.ndarray
        Rotation of the chessboard in the camera frame (with shape (N, 3, 3)).
    t_chessboard : np.ndarray
        Translation of the chessboard in the camera frame (with shape (N, 3, 1)).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the rotation matrix (with shape (3, 3)) and the translation vector (with shape (3, 1)).
        Representing the rotation and translation of the camera in the gripper frame.
    """
    R, t = cv2.calibrateHandEye(
        R_gripper, t_gripper, R_chessboard, t_chessboard, method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R, t


def main():
    parser = argparse.ArgumentParser(description="Trajectory Evaluation - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--ctrl_dt", type=float, default=0.1)
    parser.add_argument("--headless", type=int, default=1) # change this to 0 if you want to enable the mujoco viewer
    parser.add_argument("--wait_steps", type=int, default=20)
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()

    # We put the chessboard at the place where the object we want to grasp is located
    chessboard_pose = to_pose(trans=np.array([0.45, 0.2, 0.51]))
    env_config = GraspEnvConfig(
        robot=args.robot,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        wait_steps=args.wait_steps,
        chessboard_pose=chessboard_pose,
    )

    env = GraspEnv(env_config)
    env.launch()
    env.reset()

    # TODO: Complete this function to get est_result, the 4x4 matrix of the camera pose in the gripper (eef, end-effector) frame
    # since we use functions from opencv here, we use the opencv convention
    # in opencv convention, the camera pose's rotation matrix's columns represent right, down, and forward on the image respectively

    # To set the pose of robot, you can directly call env.set_eef_pose(pose) or env.reset(qpos)
    # If you use the env.set_eef_pose, you need to check the return value to know whether the pose is reachable
    # If it is not reachable, the actual eef pose might be different from the input pose and you need to skip this sample
    # env.robot_cfg.joint_init_qpos is the initial joint position, where the chessboard can be seen clearly
    # If you want to get the eef pose of it, you can call env.robot_model.fk_eef(env.robot_cfg.joint_init_qpos)

    # You can get the rgb image with obs = env.get_obs(), rgb = obs.rgb
    # Intrinsics can be accessed with env.robot_cfg.camera_cfg.intrinsics

    chessboard_corners = get_corners_in_chessboard_frame(
        CALIB_CHESSBOARD_SIZE, CALIB_GRID_SCALE
    )
    intrinsics = env.robot_cfg.camera_cfg.intrinsics
    init_eef = to_pose(*env.robot_model.fk_eef(env.robot_cfg.joint_init_qpos))

    gripper_poses = []
    chessboard_poses = []
    offsets = [
        (-0.08, -0.06, 0.00),
        (-0.05, 0.04, 0.02),
        (-0.02, -0.02, -0.02),
        (0.00, 0.06, 0.00),
        (0.03, -0.05, 0.02),
        (0.05, 0.02, -0.02),
        (0.08, -0.02, 0.00),
        (-0.06, 0.00, 0.04),
        (0.06, 0.06, 0.02),
        (0.00, -0.08, 0.03),
        (-0.03, 0.08, -0.01),
        (0.04, -0.01, 0.05),
    ]
    angle_sets = [
        (0.00, 0.00, 0.00),
        (0.18, 0.00, -0.10),
        (-0.14, 0.08, 0.06),
        (0.10, -0.12, 0.12),
        (-0.18, -0.05, -0.08),
        (0.04, 0.16, -0.14),
        (0.16, -0.06, 0.04),
        (-0.08, 0.14, 0.10),
        (0.12, 0.10, -0.06),
        (-0.12, -0.10, 0.16),
        (0.06, -0.16, -0.12),
        (-0.04, 0.06, 0.18),
    ]

    for offset, angles in zip(offsets, angle_sets):
        pose = init_eef.copy()
        delta_rot = (
            axangle2mat([1, 0, 0], angles[0])
            @ axangle2mat([0, 1, 0], angles[1])
            @ axangle2mat([0, 0, 1], angles[2])
        )
        pose[:3, :3] = init_eef[:3, :3] @ delta_rot
        pose[:3, 3] = init_eef[:3, 3] + np.array(offset)
        if not env.set_eef_pose(pose):
            continue

        obs = env.get_obs()
        succ, corners = find_chessboard_corners(obs.rgb, CALIB_CHESSBOARD_SIZE)
        if not succ:
            continue
        pnp_succ, rvec, tvec = solve_pnp_checkboard(
            chessboard_corners, corners, intrinsics
        )
        if not pnp_succ:
            continue

        rot, _ = cv2.Rodrigues(rvec)
        chessboard_poses.append(to_pose(tvec.reshape(3), rot))
        gripper_poses.append(to_pose(*env.robot_model.fk_eef(env.sim.robot_qpos)))

    if len(gripper_poses) < 3:
        raise RuntimeError(f"Only collected {len(gripper_poses)} valid calibration views")

    gripper_poses = np.stack(gripper_poses)
    chessboard_poses = np.stack(chessboard_poses)
    est_rot, est_trans = calibrate_hand_eye(
        gripper_poses[:, :3, :3],
        gripper_poses[:, :3, 3].reshape(-1, 3, 1),
        chessboard_poses[:, :3, :3],
        chessboard_poses[:, :3, 3].reshape(-1, 3, 1),
    )
    est_result = to_pose(est_trans.reshape(3), est_rot)

    env.close()

    # Get the ground truth camera pose
    gt_cam = to_pose(*env.robot_model.fk_camera(env.robot_cfg.joint_init_qpos))
    gt_eef = to_pose(*env.robot_model.fk_eef(env.robot_cfg.joint_init_qpos))
    # we use left, up, forward for extrinsics in the cfg
    # opencv use right, down, forward
    # so we need to multiply by -1 in x and y axis
    align_mat = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1.],
    ])
    gt_result = np.linalg.inv(gt_eef) @ gt_cam @ align_mat

    print("translation error (m):", np.linalg.norm(gt_result[:3, 3] - est_result[:3, 3]))
    print("rotation error (deg):", np.abs(np.rad2deg(mat2axangle(np.linalg.inv(gt_result[:3, :3]) @ est_result[:3, :3])[1])))


if __name__ == "__main__":
    main()
