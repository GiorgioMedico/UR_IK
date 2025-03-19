# IK_UR

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![ci-test](https://github.com/GiorgioMedico/UR_IK/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/UR_IK/actions/workflows/test.yml)
[![pre-commit](https://github.com/GiorgioMedico/UR_IK/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/UR_IK/actions/workflows/pre-commit.yml)

## Universal Robots Inverse Kinematics Library

A comprehensive Python library for computing forward and inverse kinematics for Universal Robots (UR) manipulators. This package provides precise kinematics calculations for the UR3e, UR5e, UR10e, and UR16e robot models using the Denavit-Hartenberg convention.

### Features

- **Forward Kinematics**: Compute end-effector pose from joint angles
- **Inverse Kinematics**: Calculate joint angles from desired end-effector pose
- **Singularity Detection**: Identify and avoid kinematic singularities
- **Trajectory Generation**: Generate continuous joint trajectories between poses
- **Multi-Robot Support**: Compatible with UR3e, UR5e, UR10e, and UR16e models
- **Well-Tested**: Comprehensive test suite with high code coverage
- **Type-Hinted**: Complete type annotations for improved code quality

### Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e .[all]
```

### Usage Examples

```python
import numpy as np
from ik_ur.dh_params import get_robot_params
from ik_ur.ik import RobotKinematics

# Initialize a UR5e robot
dh_params = get_robot_params("UR5e")
robot = RobotKinematics(dh_params)

# Forward kinematics
joint_angles = np.array([0, -np.pi/4, np.pi/2, 0, np.pi/3, 0])
end_effector_pose, transforms = robot.forward_kinematics(joint_angles)

# Inverse kinematics
solutions = robot.inverse_kinematics(end_effector_pose)

# Get joint positions for visualization
joint_positions = robot.get_joint_positions(joint_angles)

# Generate a trajectory between poses
poses = [...]  # List of poses as [x, y, z, qx, qy, qz, qw]
joint_trajectory = robot.compute_joint_trajectory(
    poses,
    initial_config=joint_angles
)
```

### Supported Robot Models

The library provides Denavit-Hartenberg parameters for the following Universal Robots models:

- **UR3e**: Compact collaborative robot with 3kg payload
- **UR5e**: Mid-sized collaborative robot with 5kg payload
- **UR10e**: Large collaborative robot with 10kg payload
- **UR16e**: Heavy-duty collaborative robot with 16kg payload

### Requirements

- Python 3.10+
- NumPy 2.0.0+
- SciPy 1.15.2+
- Matplotlib 3.10.1+ (for visualization)

### Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests

# Run tests with coverage
pytest --cov=ik_ur tests --cov-report=html
```

### Project Structure

```
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── ik_ur                  # Main package
│   ├── __init__.py        # Package initialization
│   ├── dh_params.py       # Robot parameters
│   ├── ik.py              # Kinematics algorithms
│   └── version.py         # Version information
├── tests                  # Test suite
│   ├── test_ik.py         # Kinematics tests
│   └── test_traj_ik.py    # Trajectory tests
└── examples               # Usage examples
    └── main.py            # Simple example script
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.