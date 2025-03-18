import numpy as np

import pytest
from ik_ur.dh_params import get_robot_params
from ik_ur.ik import RobotKinematics


@pytest.fixture
def ur5e_robot():
    """Fixture for a UR5e robot."""
    dh_params = get_robot_params("UR5e")
    return RobotKinematics(dh_params)


@pytest.fixture
def custom_robot():
    """Fixture for a robot with custom DH parameters."""
    dh_params = {
        "a2": -0.5,  # Link length (m)
        "a3": -0.4,  # Link length (m)
        "d1": 0.1,  # Link offset (m)
        "d4": 0.1,  # Link offset (m)
        "d5": 0.1,  # Link offset (m)
        "d6": 0.08,  # Link offset (m)
    }
    return RobotKinematics(dh_params)


def test_forward_kinematics_basic(ur5e_robot):
    """Test basic forward kinematics functionality."""
    # Simple joint configuration
    joint_angles = np.array([0, 0, 0, 0, 0, 0])

    # Calculate forward kinematics
    t06, transforms = ur5e_robot.forward_kinematics(joint_angles)

    # Check that result is a valid transformation matrix
    assert t06.shape == (4, 4)
    assert np.isclose(
        np.linalg.det(t06[:3, :3]), 1.0, atol=1e-12
    ), "Rotation matrix determinant should be 1"

    # Check that transforms contains expected keys
    assert "joints" in transforms
    assert "links" in transforms
    assert "T01" in transforms["joints"]
    assert "T06" in transforms["links"]


@pytest.mark.parametrize(
    "angles",
    [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]),
        np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]),
        np.array([-np.pi / 3, np.pi / 3, -np.pi / 3, np.pi / 3, -np.pi / 3, np.pi / 3]),
    ],
)
def test_forward_kinematics_various_configurations(ur5e_robot, angles):
    """Test forward kinematics with various joint configurations."""
    # Calculate forward kinematics
    t06, _ = ur5e_robot.forward_kinematics(angles)

    # Check that result is a valid transformation matrix
    assert t06.shape == (4, 4)
    assert np.isclose(
        np.linalg.det(t06[:3, :3]), 1.0, atol=1e-5
    ), "Rotation matrix determinant should be 1"


@pytest.mark.parametrize("seed", range(5))  # Run 5 tests with different random seeds
def test_forward_inverse_consistency(ur5e_robot, seed):
    """Test that forward kinematics followed by inverse kinematics returns consistent joint angles."""
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random joint angles as NumPy array
    original_angles = np.random.uniform(-np.pi, np.pi, 6)

    # Compute forward kinematics
    t06, _ = ur5e_robot.forward_kinematics(original_angles)

    # Compute inverse kinematics
    solutions = ur5e_robot.inverse_kinematics(t06)

    # Skip test if no solutions found (could be near singularity)
    if solutions.size == 0:
        pytest.skip("No inverse kinematics solutions found - may be near singularity")

    # Get the best solution
    best_solution = ur5e_robot.get_best_solution(solutions, original_angles)

    # Ensure we have a valid solution before proceeding
    if best_solution is None:
        pytest.fail("Could not find a best solution")

    # At this point, best_solution is guaranteed to be not None
    verified_solution = best_solution
    t06_verify, _ = ur5e_robot.forward_kinematics(verified_solution)

    # Check position error
    pos_error = np.linalg.norm(t06[:3, 3] - t06_verify[:3, 3])
    assert pos_error < 1e-3, f"Position error too large: {pos_error}"

    # Check orientation error
    ori_error = np.linalg.norm(t06[:3, :3] - t06_verify[:3, :3], "fro")
    assert ori_error < 1e-3, f"Orientation error too large: {ori_error}"


@pytest.mark.parametrize(
    "q2, q3, q4, q5, expected",
    [
        # Wrist singularity: q5 ≈ 0
        (np.pi / 2, np.pi / 2, 0, 1e-10, True),
        # Elbow singularity: q3 ≈ 0
        (np.pi / 2, 1e-10, 0, np.pi / 2, True),
        # Regular configuration, not a singularity
        (np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, False),
    ],
)
def test_singularity_detection(ur5e_robot, q2, q3, q4, q5, expected):
    """Test detection of singularities."""
    result = ur5e_robot.is_singularity(q2, q3, q4, q5)
    assert (
        result == expected
    ), f"Singularity detection for [{q2}, {q3}, {q4}, {q5}] should be {expected}"


def test_unreachable_positions(ur5e_robot):
    """Test behavior with unreachable positions."""
    # Calculate approximate maximum reach of the robot
    a2 = abs(ur5e_robot.dh_params["a2"])
    a3 = abs(ur5e_robot.dh_params["a3"])
    d5 = abs(ur5e_robot.dh_params["d5"])
    d6 = abs(ur5e_robot.dh_params["d6"])
    max_reach = a2 + a3 + d5 + d6

    # Create transformation matrix with position outside reachable workspace
    T = np.eye(4)
    T[0, 3] = max_reach * 2  # Position is twice the maximum reach

    # Try to compute inverse kinematics
    solutions = ur5e_robot.inverse_kinematics(T)

    # Should have no solutions for unreachable position
    assert solutions.size == 0, "Should not find solutions for unreachable position"


@pytest.mark.parametrize("robot_model", ["UR3e", "UR5e", "UR10e", "UR16e"])
def test_multiple_robot_models(robot_model):
    """Test with different robot models."""
    # Get parameters for the robot model
    dh_params = get_robot_params(robot_model)

    # Create robot kinematics instance
    robot = RobotKinematics(dh_params)

    # Test forward kinematics with a simple configuration
    joint_angles = np.array([0, np.pi / 4, np.pi / 4, 0, np.pi / 2, 0])
    t06, transforms = robot.forward_kinematics(joint_angles)

    # Verify transformation matrix
    assert t06.shape == (4, 4)

    # Test inverse kinematics
    solutions = robot.inverse_kinematics(t06)

    # Should find at least one solution
    assert solutions.size > 0, f"No solutions found for {robot_model}"

    # Verify a solution
    best_solution = robot.get_best_solution(solutions, joint_angles)

    # Ensure we have a valid solution before proceeding
    if best_solution is None:
        pytest.fail("Could not find a best solution")

    t06_verify, _ = robot.forward_kinematics(best_solution)

    # Check errors
    pos_error = np.linalg.norm(t06[:3, 3] - t06_verify[:3, 3])
    ori_error = np.linalg.norm(t06[:3, :3] - t06_verify[:3, :3], "fro")

    assert pos_error < 1e-3, f"Position error too large for {robot_model}: {pos_error}"
    assert ori_error < 1e-3, f"Orientation error too large for {robot_model}: {ori_error}"


def test_get_joint_positions(ur5e_robot):
    """Test getting joint positions."""
    # Simple joint configuration
    joint_angles = np.array([0, 0, 0, 0, 0, 0])

    # Get joint positions
    positions = ur5e_robot.get_joint_positions(joint_angles)

    # Check that result contains expected keys
    expected_keys = ["base", "joint1", "joint2", "joint3", "joint4", "joint5", "end_effector"]
    for key in expected_keys:
        assert key in positions, f"Joint positions missing key: {key}"
        assert positions[key].shape == (3,), f"Joint position {key} should be a 3D vector"


def test_get_best_solution():
    """Test getting the best solution."""
    # Create some test solutions
    solutions = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ]
    )

    # Current configuration
    current_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Get best solution
    best = RobotKinematics.get_best_solution(solutions, current_config)

    # Ensure we have a valid solution before proceeding
    if best is None:
        pytest.fail("Could not find a best solution")

    # First solution should be closest to current config
    assert np.array_equal(
        best, solutions[0]
    ), "Should select the solution closest to current configuration"

    # Test with empty solutions
    empty_solutions = np.array([]).reshape(0, 6)
    assert (
        RobotKinematics.get_best_solution(empty_solutions, current_config) is None
    ), "Should return None for empty solutions"

    # Test with None current configuration
    assert (
        RobotKinematics.get_best_solution(solutions, None) is None # type: ignore
    ), "Should return None if current config is None"


def test_dh_transform(ur5e_robot):
    """Test the DH transform calculation."""
    # Test with zero values (should result in identity matrix except for position)
    T = RobotKinematics.dh_transform(0, 0, 0, 0)
    expected = np.eye(4)
    assert np.allclose(T, expected, atol=1e-5), "DH transform with zeros should be identity"

    # Test with non-zero values
    theta = np.pi / 2
    d = 0.1
    a = 0.2
    alpha = np.pi / 2

    T = RobotKinematics.dh_transform(theta, d, a, alpha)

    # Verify properties of the transformation matrix
    assert T.shape == (4, 4), "Transform should be a 4x4 matrix"
    assert np.isclose(
        np.linalg.det(T[:3, :3]), 1.0, atol=1e-5
    ), "Rotation part should have determinant 1"
    assert np.allclose(T[3, :], [0, 0, 0, 1]), "Last row should be [0, 0, 0, 1]"


@pytest.mark.parametrize("verbose", [True, False])
def test_inverse_kinematics_verbose(ur5e_robot, verbose):
    """Test inverse kinematics with verbose flag."""
    # Simple joint configuration
    joint_angles = np.array([0, np.pi / 4, np.pi / 4, 0, np.pi / 2, 0])

    # Compute forward kinematics
    t06, _ = ur5e_robot.forward_kinematics(joint_angles)

    # Compute inverse kinematics with verbose flag
    solutions = ur5e_robot.inverse_kinematics(t06, verbose=verbose)

    # Should find at least one solution
    assert solutions.size > 0, "No solutions found"


def test_random_configurations(ur5e_robot):
    """Test with multiple random configurations."""
    np.random.seed(42)  # For reproducibility

    # Test 1000 random configurations
    for _ in range(1000):
        # Generate random joint angles
        joint_angles = np.random.uniform(-np.pi, np.pi, 6)

        # Avoid singularities
        if ur5e_robot.is_singularity(
            joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4]
        ):
            continue

        # Compute forward kinematics
        t06, _ = ur5e_robot.forward_kinematics(joint_angles)

        # Compute inverse kinematics
        solutions = ur5e_robot.inverse_kinematics(t06)

        # Skip if no solutions found
        if solutions.size == 0:
            continue

        # Verify a solution
        best_solution = ur5e_robot.get_best_solution(solutions, joint_angles)
        t06_verify, _ = ur5e_robot.forward_kinematics(best_solution)

        # Check errors
        pos_error = np.linalg.norm(t06[:3, 3] - t06_verify[:3, 3])
        ori_error = np.linalg.norm(t06[:3, :3] - t06_verify[:3, :3], "fro")

        assert pos_error < 1e-3, f"Position error too large: {pos_error}"
        assert ori_error < 1e-3, f"Orientation error too large: {ori_error}"
