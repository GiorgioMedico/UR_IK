"""
Robot kinematics module for 6-DOF UR manipulators.

This module provides a class for computing forward and inverse kinematics
for Universal Robots (UR) manipulators using the Denavit-Hartenberg convention.

The inverse kinematics implementation is based on the analytical approach presented in
"A General Analytical Algorithm for Collaborative Robot (cobot) with 6 Degree of Freedom (DOF)"
by Chen et al. (2017 IEEE International Conference on Applied System Innovation).
Unlike numerical/iterative methods, this analytical approach:

1. Ensures real-time computation with high accuracy
2. Provides closed-form solutions for all possible configurations
3. Enables efficient singularity detection and avoidance

The implementation handles edge cases, joint limits, and multiple solution selection
to provide robust inverse kinematics for practical robot control applications.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class RobotKinematics:
    """
    A class for computing forward and inverse kinematics for a 6-DOF robot.

    This class implements the Denavit-Hartenberg (DH) convention for robot kinematics
    and provides methods for forward kinematics, inverse kinematics, and singularity detection.

    Parameters
    ----------
    dh_params : dict[str, float]
        Dictionary with DH parameters (a2, a3, d1, d4, d5, d6)
    """

    def __init__(self, dh_params: dict[str, float]) -> None:
        """
        Initialize the RobotKinematics class with DH parameters.

        Parameters
        ----------
        dh_params : dict[str, float]
            Dictionary with DH parameters (a2, a3, d1, d4, d5, d6)
        """
        self.dh_params = dh_params
        self.eps = 1e-10  # Small value for numerical stability

    @staticmethod
    def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        Compute the Denavit-Hartenberg transformation matrix.

        Parameters
        ----------
        theta : float
            Joint angle
        d : float
            Link offset
        a : float
            Link length
        alpha : float
            Link twist

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )

    def forward_kinematics(
        self, joint_angles: np.ndarray
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """
        Compute the forward kinematics for a 6-DOF robot.

        Parameters
        ----------
        joint_angles : np.ndarray
            Array of 6 joint angles [theta1, theta2, theta3, theta4, theta5, theta6]

        Returns
        -------
        tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]
            - t06: 4x4 homogeneous transformation matrix representing the end-effector pose
            - ts: Dictionary containing all intermediate transformation matrices
        """
        # Extract joint angles
        theta1, theta2, theta3, theta4, theta5, theta6 = joint_angles

        # Extract DH parameters
        a2 = self.dh_params["a2"]
        a3 = self.dh_params["a3"]
        d1 = self.dh_params["d1"]
        d4 = self.dh_params["d4"]
        d5 = self.dh_params["d5"]
        d6 = self.dh_params["d6"]

        # Compute individual transformation matrices for each joint
        t01 = self.dh_transform(theta1, d1, 0, np.pi / 2)
        t12 = self.dh_transform(theta2, 0, a2, 0)
        t23 = self.dh_transform(theta3, 0, a3, 0)
        t34 = self.dh_transform(theta4, d4, 0, np.pi / 2)
        t45 = self.dh_transform(theta5, d5, 0, -np.pi / 2)
        t56 = self.dh_transform(theta6, d6, 0, 0)

        # Compute cumulative transformation matrices
        t02 = t01 @ t12
        t03 = t02 @ t23
        t04 = t03 @ t34
        t05 = t04 @ t45
        t06 = t05 @ t56

        # Store all transformation matrices
        ts = {
            "joints": {"T01": t01, "T12": t12, "T23": t23, "T34": t34, "T45": t45, "T56": t56},
            "links": {"T01": t01, "T02": t02, "T03": t03, "T04": t04, "T05": t05, "T06": t06},
        }

        return t06, ts

    def is_singularity(self, q2: float, q3: float, q4: float, q5: float) -> bool:
        """
        Check if a given joint configuration is at or near a singularity.

        Based on the determinant of the Jacobian:
        det(J) = s₃s₅a₂a₃(c₂a₂ + c₂₃a₃ + s₂₃₄d₅)

        Parameters
        ----------
        q2 : float
            Joint angle for joint 2
        q3 : float
            Joint angle for joint 3
        q4 : float
            Joint angle for joint 4
        q5 : float
            Joint angle for joint 5

        Returns
        -------
        bool
            True if the configuration is in a singularity, False otherwise
        """
        # Extract DH parameters
        a2 = self.dh_params["a2"]
        a3 = self.dh_params["a3"]
        d5 = self.dh_params["d5"]

        # Check for wrist singularity: s5 ≈ 0 (q5 ≈ 0 or q5 ≈ ±π)
        if np.isclose(np.sin(q5), 0.0, atol=self.eps):
            return True

        # Check for elbow singularity: s3 ≈ 0 (q3 ≈ 0 or q3 ≈ ±π)
        if np.isclose(np.sin(q3), 0.0, atol=self.eps):
            return True

        # Check for shoulder singularity
        c2 = np.cos(q2)
        c23 = np.cos(q2 + q3)
        s234 = np.sin(q2 + q3 + q4)

        last_factor = c2 * a2 + c23 * a3 + s234 * d5
        return bool(np.isclose(last_factor, 0.0, atol=self.eps))

    def inverse_kinematics(
        self, end_effector_pose: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        Compute the inverse kinematics for a 6-DOF robot,
        skipping solutions that are at or near singularities.

        Parameters
        ----------
        end_effector_pose : np.ndarray
            4x4 homogeneous transformation matrix representing the end-effector pose
        verbose : bool, optional
            If True, print details about skipped singularities, by default False

        Returns
        -------
        np.ndarray
            Array of possible joint configurations, each with 6 joint angles
        """
        # Extract DH parameters
        a2 = self.dh_params["a2"]
        a3 = self.dh_params["a3"]
        d1 = self.dh_params["d1"]
        d4 = self.dh_params["d4"]
        d5 = self.dh_params["d5"]
        d6 = self.dh_params["d6"]

        # Extract position and orientation from transformation matrix
        nx = end_effector_pose[0, 0]
        ny = end_effector_pose[1, 0]

        ox = end_effector_pose[0, 1]
        oy = end_effector_pose[1, 1]

        ax = end_effector_pose[0, 2]
        ay = end_effector_pose[1, 2]
        az = end_effector_pose[2, 2]

        px = end_effector_pose[0, 3]
        py = end_effector_pose[1, 3]
        pz = end_effector_pose[2, 3]

        # Solutions list
        solutions = []

        # 1. Calculate joint angle q1 (2 solutions)
        # Calculate terms for q1
        term1 = d6 * ay - py
        term2 = px - d6 * ax

        # Calculate discriminant for square root
        discriminant = term1**2 + term2**2 - d4**2

        if discriminant < -self.eps:
            if verbose:
                print("Warning: Target position appears to be unreachable for q1 calculation")
            return np.array([])  # No solutions possible

        discriminant = max(0.0, discriminant)  # Ensure non-negative for sqrt
        sqrt_term = np.sqrt(discriminant)

        # Calculate the angle terms
        term3 = np.arctan2(term1, term2)

        # First solution (positive square root)
        theta1_pos = np.arctan2(d4, sqrt_term)
        q1_1 = theta1_pos - term3

        # Second solution (negative square root)
        theta1_neg = np.arctan2(d4, -sqrt_term)
        q1_2 = theta1_neg - term3

        q1_solutions = [q1_1, q1_2]

        # Try all q1 solutions
        for q1 in q1_solutions:
            s1 = np.sin(q1)
            c1 = np.cos(q1)

            # Calculate terms for q5
            u_s = (nx * s1 - ny * c1) ** 2 + (ox * s1 - oy * c1) ** 2

            if u_s < -self.eps:
                if verbose:
                    print("Warning: Target position appears to be unreachable for q5 calculation")
                continue  # Skip this configuration

            numerator = np.sqrt(max(0.0, u_s))
            denominator = ax * s1 - ay * c1

            # Check for potential wrist singularity
            if np.isclose(numerator, 0.0, atol=self.eps) and np.isclose(
                denominator, 0.0, atol=self.eps
            ):
                if verbose:
                    print(f"Skipping wrist singularity at q1 = {q1:.4f}")
                continue  # Skip this configuration

            # Two solutions for q5
            q5_1 = np.arctan2(numerator, denominator)
            q5_2 = np.arctan2(-numerator, denominator)

            q5_solutions = [q5_1, q5_2]

            # Try all q5 solutions
            for q5 in q5_solutions:
                s5 = np.sin(q5)

                # Check for wrist singularity (s5 ≈ 0)
                if np.isclose(s5, 0.0, atol=self.eps):
                    if verbose:
                        print(f"Skipping wrist singularity at q5 = {q5:.4f}")
                    continue  # Skip this singularity

                # 3. Calculate joint angle q6 (1 solution for each q1, q5 pair)
                q6 = np.arctan2(-((ox * s1 - oy * c1) / s5), (nx * s1 - ny * c1) / s5)

                # 4. Calculate joint angle q234 (1 solution for each q1, q5 pair)
                q234 = np.arctan2(-az / s5, -(ax * c1 + ay * s1) / s5)

                s234 = np.sin(q234)
                c234 = np.cos(q234)

                # 5. Calculate joint angle q2 (2 solutions for each q1, q5, q6, q234 set)
                # Define a and b
                a_val = px * c1 + py * s1 - d5 * s234 + d6 * s5 * c234
                b_val = pz - d1 + d5 * c234 + d6 * s5 * s234

                # Calculate η (eta)
                eta = np.sqrt(max(0.0, a_val**2 + b_val**2))

                # Calculate the numerator term
                numerator = a_val**2 + b_val**2 + a2**2 - a3**2

                # Check if the position is reachable
                if abs(numerator / (2 * a2 * eta)) > 1 + self.eps:
                    if verbose:
                        print("Target position appears to be unreachable for q2")
                    continue  # Skip this configuration

                # Ensure value is within valid range
                term1 = np.clip(numerator / (2 * a2 * eta), -1.0, 1.0)

                # Calculate the two solutions
                q2_1 = np.arctan2(term1, np.sqrt(1 - term1**2)) - np.arctan2(a_val, b_val)
                q2_2 = np.arctan2(term1, -np.sqrt(1 - term1**2)) - np.arctan2(a_val, b_val)

                q2_solutions = [q2_1, q2_2]

                # Try all q2 solutions
                for q2 in q2_solutions:
                    s2 = np.sin(q2)
                    c2 = np.cos(q2)

                    # 6. Calculate joint angle q3
                    pz_d1_term = pz - d1 + d5 * c234 + d6 * s5 * s234
                    px_py_term = px * c1 + py * s1 - d5 * s234 + d6 * s5 * c234

                    # Calculate the terms for q23
                    numer = (pz_d1_term - a2 * s2) / a3
                    denom = (px_py_term - a2 * c2) / a3

                    # Compute q23
                    q23 = np.arctan2(numer, denom)

                    # Compute q3
                    q3 = q23 - q2

                    # Check for elbow singularity
                    if np.isclose(np.sin(q3), 0.0, atol=self.eps):
                        if verbose:
                            print(f"Skipping elbow singularity at q3 = {q3:.4f}")
                        continue  # Skip this singularity

                    # 7. Calculate joint angle q4
                    q4 = q234 - q23

                    # Check for shoulder singularity
                    shoulder_factor = c2 * a2 + np.cos(q2 + q3) * a3 + np.sin(q2 + q3 + q4) * d5
                    if np.isclose(shoulder_factor, 0.0, atol=self.eps):
                        if verbose:
                            print("Skipping shoulder singularity")
                        continue  # Skip this singularity

                    # Final check for any singularity
                    if self.is_singularity(q2, q3, q4, q5):
                        if verbose:
                            print("Skipping detected singularity")
                        continue

                    # Normalize angles to [-π, π] range
                    solution = np.array(
                        [
                            np.mod(q1 + np.pi, 2 * np.pi) - np.pi,
                            np.mod(q2 + np.pi, 2 * np.pi) - np.pi,
                            np.mod(q3 + np.pi, 2 * np.pi) - np.pi,
                            np.mod(q4 + np.pi, 2 * np.pi) - np.pi,
                            np.mod(q5 + np.pi, 2 * np.pi) - np.pi,
                            np.mod(q6 + np.pi, 2 * np.pi) - np.pi,
                        ]
                    )

                    solutions.append(solution)

        # Verify solutions using forward kinematics
        error_threshold = 1e-3
        verified_solutions = []

        if verbose:
            print(f"\nNumber of candidate solutions found: {len(solutions)}")

        for i, solution in enumerate(solutions):
            # Verify solution using forward kinematics
            t06_verify, _ = self.forward_kinematics(solution)

            # Compute position error
            pos_error = np.linalg.norm(end_effector_pose[0:3, 3] - t06_verify[0:3, 3])

            # Compute orientation error (using Frobenius norm)
            ori_error = np.linalg.norm(end_effector_pose[0:3, 0:3] - t06_verify[0:3, 0:3], "fro")

            # Only keep solutions with acceptable error
            if pos_error <= error_threshold and ori_error <= error_threshold:
                verified_solutions.append(solution)

                # Print verification details if verbose
                if verbose:
                    print(f"\nSolution {i + 1}:")
                    print("Joint angles (rad):", solution)
                    print(f"Position error (m): {pos_error:.6f}")
                    print(f"Orientation error: {ori_error:.6f}")
            elif verbose:
                print(f"\nRejected Solution {i + 1} (exceeds error threshold):")
                print("Joint angles (rad):", solution)
                print(f"Position error (m): {pos_error:.6f}")
                print(f"Orientation error: {ori_error:.6f}")

        if verbose:
            print(f"\nNumber of verified solutions: {len(verified_solutions)}")

        return np.array(verified_solutions)

    def get_joint_positions(self, joint_angles: np.ndarray) -> dict[str, np.ndarray]:
        """
        Calculate the positions of all joints given a joint configuration.

        Parameters
        ----------
        joint_angles : np.ndarray
            Array of 6 joint angles [theta1, theta2, theta3, theta4, theta5, theta6]

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with joint positions, where keys are joint names and
            values are 3D position vectors
        """
        # Get transformation matrices
        _, ts = self.forward_kinematics(joint_angles)

        # Extract joint positions from transformation matrices
        return {
            "base": np.array([0, 0, 0]),
            "joint1": ts["links"]["T01"][0:3, 3],
            "joint2": ts["links"]["T02"][0:3, 3],
            "joint3": ts["links"]["T03"][0:3, 3],
            "joint4": ts["links"]["T04"][0:3, 3],
            "joint5": ts["links"]["T05"][0:3, 3],
            "end_effector": ts["links"]["T06"][0:3, 3],
        }

    @staticmethod
    def get_best_solution(solutions: np.ndarray, current_config: np.ndarray) -> np.ndarray | None:
        """
        Select the best solution from an array of possible joint configurations.

        The best solution is selected based on minimum joint movement from the
        current configuration, or minimum joint values if no current configuration
        is provided.

        Parameters
        ----------
        solutions : np.ndarray
            Array of possible joint configurations
        current_config : np.ndarray
            Current joint configuration

        Returns
        -------
        np.ndarray | None
            Best joint configuration, or None if no solutions are provided
        """
        if solutions.size == 0 or current_config is None:
            return None

        # Calculate joint movement for each solution using vectorized operations
        joint_movements = np.sum(np.abs(solutions - current_config), axis=1)

        # Select solution with minimum joint movement
        best_idx = np.argmin(joint_movements)
        return solutions[best_idx]

    def compute_joint_trajectory(
        self,
        poses: np.ndarray,
        joint_limits: np.ndarray | None = None,
        initial_config: np.ndarray | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, bool, str]:
        """
        Compute a continuous joint trajectory for a sequence of poses and detect jumps.

        This method computes inverse kinematics for each pose in the sequence and
        ensures that the resulting joint trajectory is continuous by minimizing
        jumps between consecutive configurations. It also enforces joint limits.
        It checks for jumps in the trajectory and returns a flag and warning message
        if jumps are detected.

        Parameters
        ----------
        poses : np.ndarray
            Array of poses, each as [x, y, z, qx, qy, qz, qw] where:
            - x, y, z: Position coordinates
            - qx, qy, qz, qw: Quaternion orientation
        joint_limits : np.ndarray | None, optional
            Joint limits as [min1, max1, min2, max2, min3, max3, min4, max4, min5, max5,
                            min6, max6].
            If None, defaults to [-2π, 2π] for joints 1-6.
        initial_config : np.ndarray | None, optional
            Initial joint configuration as [j1, j2, j3, j4, j5, j6].
            If None, defaults to [0, 0, 0, 0, 0, 0].
        verbose : bool, optional
            If True, print detailed information, by default False

        Returns
        -------
        tuple[np.ndarray, bool, str]
            - Joint trajectory as an array of joint angles for each pose
            - Boolean indicating if the trajectory is valid (no jumps)
            - Warning message if jumps were detected, empty string otherwise
        """
        # Convert inputs to numpy arrays
        poses = np.asarray(poses)

        # Set default joint limits if not provided
        if joint_limits is None:
            joint_limits = np.array(
                [
                    -2 * np.pi,
                    2 * np.pi,  # Joint 1
                    -2 * np.pi,
                    2 * np.pi,  # Joint 2
                    -2 * np.pi,
                    2 * np.pi,  # Joint 3
                    -2 * np.pi,
                    2 * np.pi,  # Joint 4
                    -2 * np.pi,
                    2 * np.pi,  # Joint 5
                    -2 * np.pi,
                    2 * np.pi,  # Joint 6
                ]
            )
        else:
            joint_limits = np.asarray(joint_limits)

        # Set default initial configuration if not provided
        initial_config = np.zeros(6) if initial_config is None else np.asarray(initial_config)

        jump_threshold = np.pi / 2

        # Initialize trajectory array
        trajectory = np.zeros((len(poses), 6))

        # Set initial configuration
        current_config = initial_config.copy()

        # Flag to track if jumps were detected
        jumps_detected = False
        jump_warnings = []

        # Process each pose
        for i, pose in enumerate(poses):
            if verbose:
                print(f"Processing pose {i + 1}/{len(poses)}")

            # Convert pose to transformation matrix
            transform = self._pose_to_transform(pose)

            # Compute inverse kinematics
            solutions = self.inverse_kinematics(transform, verbose=verbose)

            # If no solutions found, use the last valid configuration
            if solutions.size == 0:
                if verbose:
                    print(f"No solutions found for pose {i + 1}. Using last valid configuration.")
                trajectory[i] = current_config
                continue

            # Get the best solution (closest to current configuration)
            if verbose:
                print("Shape of IK solutions : ", solutions.shape)

            best_solution = self.get_best_solution(solutions, current_config)

            if best_solution is None:
                if verbose:
                    print(
                        f"Best solution not found for pose {i + 1}. Using last valid configuration."
                    )
                trajectory[i] = current_config
                continue

            # Ensure joint continuity
            continuous_solution = self._make_continuous(best_solution, current_config, joint_limits)

            # Check for jumps if not the first pose
            if i > 0:
                # Calculate absolute differences between consecutive configurations
                diffs = np.abs(continuous_solution - trajectory[i - 1])

                # Check if any joint difference exceeds the threshold
                if np.any(diffs > jump_threshold):
                    jumps_detected = True
                    jump_joint_indices = np.where(diffs > jump_threshold)[0]
                    jump_joints = [f"joint{j + 1}" for j in jump_joint_indices]
                    jump_sizes = [f"{diffs[j]:.4f} rad" for j in jump_joint_indices]

                    jump_warning = f"Jump detected at pose {i + 1}: {', '.join(jump_joints)} moved by {', '.join(jump_sizes)}"  # noqa: E501
                    jump_warnings.append(jump_warning)

                    if verbose:
                        print(jump_warning)

            # Store the solution
            trajectory[i] = continuous_solution

            # Update current configuration
            current_config = continuous_solution.copy()

        # Create warning message
        warning_message = ""
        if jumps_detected:
            warning_message = "WARNING: Jumps detected in joint trajectory:\n" + "\n".join(
                jump_warnings
            )
            if verbose:
                print(warning_message)

        return trajectory, not jumps_detected, warning_message

    @staticmethod
    def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
        """
        Convert a pose [x, y, z, qx, qy, qz, qw] to a 4x4 homogeneous transformation matrix.

        Parameters
        ----------
        pose : np.ndarray
            Pose as [x, y, z, qx, qy, qz, qw] where:
            - x, y, z: Position coordinates
            - qx, qy, qz, qw: Quaternion orientation

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix
        """

        x, y, z, qx, qy, qz, qw = pose

        # Create rotation matrix from quaternion using scipy
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        rotation_matrix = rotation.as_matrix()

        # Create homogeneous transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = rotation_matrix
        transform[0:3, 3] = [x, y, z]

        return transform

    @staticmethod
    def _make_continuous(
        current_solution: np.ndarray, previous_solution: np.ndarray, joint_limits: np.ndarray
    ) -> np.ndarray:
        """
        Make joint angles continuous by shifting values by multiples of 2π
        when there are jumps, ensuring they fall within joint limits.

        Parameters
        ----------
        current_solution : np.ndarray
            Current joint angles [j1, j2, j3, j4, j5, j6]
        previous_solution : np.ndarray
            Previous joint angles [j1, j2, j3, j4, j5, j6]
        joint_limits : np.ndarray
            Joint limits as [min1, max1, min2, max2, min3, max3, min4, max4, min5, max5, min6, max6]

        Returns
        -------
        np.ndarray
            Continuous joint angles [j1, j2, j3, j4, j5, j6] within the specified joint limits
        """
        continuous_solution = current_solution.copy()

        for j in range(6):
            # Get the joint limits for this joint
            min_limit_joint = joint_limits[2 * j]
            max_limit_joint = joint_limits[2 * j + 1]

            # Calculate the difference between current and previous solution
            diff = continuous_solution[j] - previous_solution[j]

            # If difference is small, no need to adjust
            if abs(diff) <= np.pi:
                continue

            # Calculate number of 2π rotations needed to minimize the jump
            num_rotations = round(diff / (2 * np.pi))

            if num_rotations == 0:
                continue

            # Calculate the adjusted value
            adjusted_value = continuous_solution[j] - num_rotations * 2 * np.pi

            # Check if the adjusted value is within joint limits
            if min_limit_joint <= adjusted_value <= max_limit_joint:
                continuous_solution[j] = adjusted_value
            else:
                # Try other adjustment directions if needed
                for offset in [-2, -1, 1, 2]:
                    if offset == -num_rotations:
                        continue  # Skip the already tried adjustment

                    test_value = continuous_solution[j] - offset * 2 * np.pi
                    if min_limit_joint <= test_value <= max_limit_joint:

                        continuous_solution[j] = test_value
                        break

        return continuous_solution
