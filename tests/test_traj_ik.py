import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ik_ur.dh_params import get_robot_params
from ik_ur.ik import RobotKinematics
from typing import List


@pytest.fixture
def ur5e_robot() -> RobotKinematics:
    """
    Fixture for a UR5e robot.

    Returns
    -------
    RobotKinematics
        A RobotKinematics instance initialized with UR5e parameters
    """
    dh_params = get_robot_params("UR5e")
    return RobotKinematics(dh_params)


def test_single_joint_rotation_trajectory(ur5e_robot: RobotKinematics) -> None:
    """
    Test trajectory generation for a single joint rotating from 0 to 2π while keeping all other joints fixed.
    This test verifies that:
    1. The trajectory planner can handle a full joint rotation (0 to 2π)
    2. The trajectory maintains continuity through the rotation
    3. The motion is smooth without large jumps at the angle wrap-around point
    """
    # Number of steps for the full rotation
    num_steps: int = 36  # 10-degree increments

    # Initial configuration - we'll use joint 4 for our test
    initial_config: np.ndarray = np.array([0.0, -np.pi/4, np.pi/2, 0.0, np.pi/3, 0.0])
    final_config: np.ndarray = np.array([0.0, -np.pi/4, np.pi/2, 2*np.pi, np.pi/3, 0.0])

    # Create intermediate joint configurations by linearly interpolating joint 4
    joint_configs: List[np.ndarray] = []
    for step in range(num_steps + 1):  # +1 to include the final configuration
        t: float = step / num_steps
        interpolated_config: np.ndarray = initial_config + t * (final_config - initial_config)
        joint_configs.append(interpolated_config)
    joint_configs_array: np.ndarray = np.array(joint_configs)

    # Get end-effector poses by applying forward kinematics
    poses: List[np.ndarray] = []
    for config in joint_configs_array:
        # Get end-effector pose with forward kinematics
        ee_pose: np.ndarray
        joint_transforms: dict[str, dict[str, np.ndarray]]
        ee_pose, joint_transforms = ur5e_robot.forward_kinematics(config)

        # Convert to [x, y, z, qx, qy, qz, qw] format for IK
        position: np.ndarray = ee_pose[:3, 3]
        rotation: Rotation = Rotation.from_matrix(ee_pose[:3, :3])
        quaternion: np.ndarray = rotation.as_quat()  # returns [qx, qy, qz, qw]

        # Create pose in expected format [x, y, z, qx, qy, qz, qw]
        pose: np.ndarray = np.concatenate([position, quaternion])
        poses.append(pose)
    poses_array: np.ndarray = np.array(poses)

    # Compute joint trajectory from poses using inverse kinematics
    computed_trajectory: np.ndarray = ur5e_robot.compute_joint_trajectory(
        poses_array,
        initial_config=initial_config,
        verbose=False
    )

    # Check that computed trajectory matches original joint configurations
    for j in range(6):
        assert np.all(np.abs(joint_configs_array[:, j] - computed_trajectory[:, j]) < 1e-2)


def test_single_joint_rotation_negative_trajectory(ur5e_robot: RobotKinematics) -> None:
    """
    Test trajectory generation for a single joint rotating from 0 to -2π while keeping all other joints fixed.
    This test verifies that:
    1. The trajectory planner can handle a full joint rotation (0 to -2π)
    2. The trajectory maintains continuity through the rotation
    3. The motion is smooth without large jumps at the angle wrap-around point
    """
    # Number of steps for the full rotation
    num_steps: int = 36  # 10-degree increments

    # Initial configuration - we'll use joint 4 for our test
    initial_config: np.ndarray = np.array([0.0, -np.pi/4, -np.pi/4, 0.0, np.pi/3, 0.0])
    final_config: np.ndarray = np.array([0.0, -np.pi/4, -np.pi/4, -2*np.pi, np.pi/3, 0.0])

    # Create intermediate joint configurations by linearly interpolating joint 4
    joint_configs: List[np.ndarray] = []
    for step in range(num_steps + 1):  # +1 to include the final configuration
        t: float = step / num_steps
        interpolated_config: np.ndarray = initial_config + t * (final_config - initial_config)
        joint_configs.append(interpolated_config)
    joint_configs_array: np.ndarray = np.array(joint_configs)

    # Get end-effector poses by applying forward kinematics
    poses: List[np.ndarray] = []
    for config in joint_configs_array:
        # Get end-effector pose with forward kinematics
        ee_pose: np.ndarray
        joint_transforms: dict[str, dict[str, np.ndarray]]
        ee_pose, joint_transforms = ur5e_robot.forward_kinematics(config)

        # Convert to [x, y, z, qx, qy, qz, qw] format for IK
        position: np.ndarray = ee_pose[:3, 3]
        rotation: Rotation = Rotation.from_matrix(ee_pose[:3, :3])
        quaternion: np.ndarray = rotation.as_quat()  # returns [qx, qy, qz, qw]

        # Create pose in expected format [x, y, z, qx, qy, qz, qw]
        pose: np.ndarray = np.concatenate([position, quaternion])
        poses.append(pose)
    poses_array: np.ndarray = np.array(poses)

    # Compute joint trajectory from poses using inverse kinematics
    computed_trajectory: np.ndarray = ur5e_robot.compute_joint_trajectory(
        poses_array,
        initial_config=initial_config,
        verbose=False
    )

    # Check that computed trajectory matches original joint configurations
    for j in range(6):
        assert np.all(np.abs(joint_configs_array[:, j] - computed_trajectory[:, j]) < 1e-2)