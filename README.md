# IK_UR: Universal Robots Inverse Kinematics Library

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![ci-test](https://github.com/GiorgioMedico/UR_IK/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/UR_IK/actions/workflows/test.yml)
[![pre-commit](https://github.com/GiorgioMedico/UR_IK/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/UR_IK/actions/workflows/pre-commit.yml)

## Overview

IK_UR is a high-performance Python library implementing analytical forward and inverse kinematics for Universal Robots (UR) manipulators. This package provides precise, real-time kinematic calculations for the UR3e, UR5e, UR10e, and UR16e robot models using the Denavit-Hartenberg convention.

Unlike numerical/iterative methods, this analytical approach:
1. Ensures real-time computation with high accuracy
2. Provides closed-form solutions for all possible configurations
3. Enables efficient singularity detection and avoidance

The implementation handles edge cases, joint limits, and multiple solution selection to provide robust inverse kinematics for practical robot control applications.

## Features

- **Forward Kinematics**: Compute end-effector pose from joint angles with high precision
- **Analytical Inverse Kinematics**: Calculate joint angles from desired end-effector pose
- **Singularity Detection**: Identify and avoid kinematic singularities with built-in detection
- **Joint Trajectory Generation**: Generate continuous joint trajectories between poses
- **Multi-Robot Support**: Compatible with UR3e, UR5e, UR10e, and UR16e models with accurate DH parameters
- **Comprehensive Test Suite**: Fully tested for accuracy and stability (10,000+ test cases)
- **Type-Hinted**: Complete type annotations for improved code quality and IDE support

## Installation

### Basic Installation

```bash
# Install from source
pip install -e .
```

## Usage Examples

### Basic Forward and Inverse Kinematics

```python
import numpy as np
from ik_ur.dh_params import get_robot_params
from ik_ur.ik import RobotKinematics

# Initialize a UR5e robot
dh_params = get_robot_params("UR5e")
robot = RobotKinematics(dh_params)

# Forward kinematics - calculate end-effector pose from joint angles
joint_angles = np.array([0, -np.pi/4, np.pi/2, 0, np.pi/3, 0])
end_effector_pose, transforms = robot.forward_kinematics(joint_angles)

# Inverse kinematics - calculate joint angles from end-effector pose
solutions = robot.inverse_kinematics(end_effector_pose)

# Get the best solution (closest to initial configuration)
best_solution = robot.get_best_solution(solutions, joint_angles)

# Get joint positions for visualization
joint_positions = robot.get_joint_positions(joint_angles)
```

### Trajectory Generation

```python
from scipy.spatial.transform import Rotation

# Create a sequence of poses (e.g., for a circular trajectory)
poses = []
for i in range(num_points):
    # Create a new pose (position and orientation)
    position = np.array([x, y, z])
    rotation = Rotation.from_euler('xyz', [rx, ry, rz])
    quaternion = rotation.as_quat()
    
    # Combine position and orientation [x, y, z, qx, qy, qz, qw]
    pose = np.concatenate([position, quaternion])
    poses.append(pose)

# Generate a continuous joint trajectory
joint_trajectory, is_valid, warning = robot.compute_joint_trajectory(
    poses,
    initial_config=joint_angles
)

# Check if the trajectory is valid (no large jumps)
if not is_valid:
    print(f"Warning: {warning}")
```

## Supported Robot Models

The library provides accurate Denavit-Hartenberg parameters for the following Universal Robots models:

| Robot Model | Payload | Description |
|-------------|---------|-------------|
| **UR3e**    | 3kg     | Compact collaborative robot for light assembly and limited workspaces |
| **UR5e**    | 5kg     | Mid-sized collaborative robot for machine tending and general automation |
| **UR10e**   | 10kg    | Large collaborative robot for packaging, palletizing, and heavier tasks |
| **UR16e**   | 16kg    | Heavy-duty collaborative robot for industrial applications requiring high payloads |

## Implementation Details

The inverse kinematics implementation is based on the analytical approach presented in "A General Analytical Algorithm for Collaborative Robot (cobot) with 6 Degree of Freedom (DOF)" by Chen et al. (2017 IEEE International Conference on Applied System Innovation).

Key features of the implementation:
- **Analytical Solutions**: Provides closed-form solutions rather than iterative approximations
- **Singularity Handling**: Robust detection and handling of singularity configurations
- **Multiple Solutions**: Returns all valid solutions, allowing for optimal path selection
- **Continuity Preservation**: Ensures joint angle continuity across trajectory points
- **Error Validation**: Verifies solutions with forward kinematics for high accuracy

## Requirements

- Python 3.10+
- NumPy 2.0.0+
- SciPy 1.15.2+
- Matplotlib 3.10.1+ (for visualization)

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests

# Run tests with coverage
pytest --cov=ik_ur tests --cov-report=html

# Run pre-commit hooks
pre-commit run --all-files
```

## Project Structure

```
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── ik_ur                  # Main package
│   ├── __init__.py        # Package initialization
│   ├── dh_params.py       # Robot DH parameters
│   ├── ik.py              # Kinematics algorithms
│   └── version.py         # Version information
├── tests                  # Test suite
│   ├── test_ik.py         # Forward/inverse kinematics tests
│   └── test_traj_ik.py    # Trajectory generation tests
└── examples               # Usage examples
    └── main.py            # Example application
```

## Citation

If you use this library in your research, please cite:

```
@software{IK_UR,
  author = {Medico, Giorgio},
  title = {IK_UR: Universal Robots Inverse Kinematics Library},
  url = {https://github.com/GiorgioMedico/UR_IK},
  year = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.