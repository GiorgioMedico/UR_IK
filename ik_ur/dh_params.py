"""
DH parameters module for Universal Robots series.

This module provides Denavit-Hartenberg parameters for different Universal Robots
models (UR3e, UR5e, UR10e, UR16e), which can be used with the RobotKinematics class.
"""

from typing import Literal


# DH parameters for UR3e robot
UR3E_PARAMS: dict[str, float] = {
    "a2": -0.24355,  # Link length (m)
    "a3": -0.2132,  # Link length (m)
    "d1": 0.15185,  # Link offset (m)
    "d4": 0.13105,  # Link offset (m)
    "d5": 0.08535,  # Link offset (m)
    "d6": 0.0921,  # Link offset (m)
}

# DH parameters for UR5e robot
UR5E_PARAMS: dict[str, float] = {
    "a2": -0.425,  # Link length (m)
    "a3": -0.3922,  # Link length (m)
    "d1": 0.1625,  # Link offset (m)
    "d4": 0.1333,  # Link offset (m)
    "d5": 0.0997,  # Link offset (m)
    "d6": 0.0996,  # Link offset (m)
}

# DH parameters for UR10e robot
UR10E_PARAMS: dict[str, float] = {
    "a2": -0.6127,  # Link length (m)
    "a3": -0.57155,  # Link length (m)
    "d1": 0.1807,  # Link offset (m)
    "d4": 0.17415,  # Link offset (m)
    "d5": 0.11985,  # Link offset (m)
    "d6": 0.11655,  # Link offset (m)
}

# DH parameters for UR16e robot
UR16E_PARAMS: dict[str, float] = {
    "a2": -0.4784,  # Link length (m)
    "a3": -0.36,  # Link length (m)
    "d1": 0.1807,  # Link offset (m)
    "d4": 0.17415,  # Link offset (m)
    "d5": 0.11985,  # Link offset (m)
    "d6": 0.11655,  # Link offset (m)
}

# Map of robot model names to their DH parameters
ROBOT_DH_PARAMS = {
    "UR3e": UR3E_PARAMS,
    "UR5e": UR5E_PARAMS,
    "UR10e": UR10E_PARAMS,
    "UR16e": UR16E_PARAMS,
}

RobotModelType = Literal["UR3e", "UR5e", "UR10e", "UR16e"]


def get_robot_params(robot_model: RobotModelType | str) -> dict[str, float]:
    """
    Get the DH parameters for a specific robot model.

    Parameters
    ----------
    robot_model : str
        The robot model name ("UR3e", "UR5e", "UR10e", or "UR16e")

    Returns
    -------
    Dict[str, float]
        Dictionary containing the DH parameters for the specified robot model

    Raises
    ------
    ValueError
        If the specified robot model is not supported
    """
    if robot_model not in ROBOT_DH_PARAMS:
        supported_models = list(ROBOT_DH_PARAMS.keys())
        raise ValueError(
            f"Robot model '{robot_model}' not supported. "
            f"Supported models are: {supported_models}"
        )

    return ROBOT_DH_PARAMS[robot_model]
