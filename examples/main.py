import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from ik_ur.dh_params import get_robot_params
from ik_ur.ik import RobotKinematics
from ik_ur.version import __version__
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


if __name__ == "__main__":
    print(f"UR_IK Package Version: {__version__}\n")

    # Initialize a UR5e robot
    robot_model = "UR5e"
    print(f"Using robot model: {robot_model}")

    dh_params = get_robot_params(robot_model)
    robot = RobotKinematics(dh_params)

    # Define initial joint configuration
    initial_angles = np.array([0, -np.pi / 4, np.pi / 2, 0, np.pi / 3, 0])
    print("\nInitial joint configuration (rad):", initial_angles)

    # Compute forward kinematics
    end_effector_pose, _ = robot.forward_kinematics(initial_angles)
    print("\nEnd-effector position (m):", end_effector_pose[:3, 3])

    # Extract rotation as Euler angles for display
    rotation = Rotation.from_matrix(end_effector_pose[:3, :3])
    euler_angles = rotation.as_euler("xyz", degrees=True)
    print("End-effector orientation (Euler angles, deg):", euler_angles)

    # Compute inverse kinematics
    solutions = robot.inverse_kinematics(end_effector_pose)
    print(f"\nFound {len(solutions)} inverse kinematics solutions")

    if len(solutions) > 0:
        # Get best solution (closest to initial configuration)
        best_solution = robot.get_best_solution(solutions, initial_angles)
        print("Best IK solution (rad):", best_solution)
        # Verify solution using forward kinematics
        ee_verify, _ = robot.forward_kinematics(best_solution)
        pos_error = np.linalg.norm(end_effector_pose[:3, 3] - ee_verify[:3, 3])
        print(f"Position error: {pos_error:.6f} m")

    # Create a circular trajectory in the XZ plane
    print("\nGenerating a circular trajectory in the XZ plane...")
    radius = 0.2  # 20cm radius
    center = end_effector_pose[:3, 3].copy()
    num_points = 10

    # Create poses for a circular trajectory
    poses = []
    for i in range(num_points):
        angle = 2 * np.pi * i / (num_points - 1)

        # Create a new pose (circular motion in XZ plane)
        new_pose = end_effector_pose.copy()
        new_pose[0, 3] = center[0] + radius * np.cos(angle)
        new_pose[2, 3] = center[2] + radius * np.sin(angle)

        # Convert to [x, y, z, qx, qy, qz, qw] format
        position = new_pose[:3, 3]
        rotation = Rotation.from_matrix(new_pose[:3, :3])
        quaternion = rotation.as_quat()
        poses.append(np.concatenate([position, quaternion]))

    poses = np.array(poses)

    # Generate trajectory
    print("\nGenerating joint trajectory for circular motion...")
    joint_trajectory, is_valid, warning = robot.compute_joint_trajectory(
        poses, initial_config=initial_angles
    )

    print(f"Trajectory has {len(joint_trajectory)} points")
    print(f"Trajectory is valid: {is_valid}")
    if warning:
        print(f"Warning: {warning}")

    # Print the first and last points in the trajectory
    print("\nFirst joint configuration in trajectory (rad):", joint_trajectory[0])
    print("Last joint configuration in trajectory (rad):", joint_trajectory[-1])

    print("\nVisualizing robot and trajectory...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get end-effector positions for each point in the trajectory
    ee_positions_list = []
    for joint_config in joint_trajectory:
        ee_pose, _ = robot.forward_kinematics(joint_config)
        ee_positions_list.append(ee_pose[:3, 3])

    # Convert list to numpy array for proper indexing
    ee_positions = np.array(ee_positions_list)

    # Explicitly extract the columns to avoid tuple indexing errors
    x_coords = ee_positions[:, 0]
    y_coords = ee_positions[:, 1]
    z_coords = ee_positions[:, 2]

    # Plot the circular trajectory of the end-effector
    ax.plot(
        x_coords,
        y_coords,
        z_coords,
        "b-",
        linewidth=2,
        label="End-Effector Path",
    )

    # Mark the start and end points
    ax.scatter(
        ee_positions[0, 0],
        ee_positions[0, 1],
        ee_positions[0, 2],
        color="green",
        s=100,
        label="Start",
    )
    ax.scatter(
        ee_positions[-1, 0],
        ee_positions[-1, 1],
        ee_positions[-1, 2],
        color="red",
        s=100,
        label="End",
    )

    # Configure the plot
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{robot_model} End-Effector Trajectory")
    ax.legend()

    plt.tight_layout()
    plt.show()
