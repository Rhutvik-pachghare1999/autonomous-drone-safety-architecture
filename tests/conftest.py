"""
ASU Robotics — AISP Pytest Configuration.

Provides fixtures for all test suites.
"""
import numpy as np
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return 0.0


@pytest.fixture
def sample_state():
    """Sample drone state vector [pos(3), vel(3), quat(4), gyro_bias(3), accel_bias(3)]."""
    return np.array([
        0.0, 0.0, -50.0,       # position NED [m]
        1.0, 0.0, 0.0,         # velocity [m/s]
        1.0, 0.0, 0.0, 0.0,    # quaternion [w, x, y, z]
        0.0, 0.0, 0.0,         # gyro bias [rad/s]
        0.0, 0.0, 0.0,         # accel bias [m/s²]
    ])


@pytest.fixture
def sample_imu_data():
    """Sample IMU measurement."""
    return {
        "accel": np.array([0.1, -0.05, -9.81]),  # m/s²
        "gyro": np.array([0.01, -0.005, 0.002]),  # rad/s
        "dt": 0.01,
    }


@pytest.fixture
def sample_gps_data():
    """Sample GPS measurement."""
    return {
        "position": np.array([0.5, 0.1, -49.8]),
        "covariance": np.diag([2.5, 2.5, 5.0]),
    }


@pytest.fixture
def sample_waypoints():
    """Sample mission waypoints."""
    return [
        np.array([0.0, 0.0, -50.0]),
        np.array([100.0, 0.0, -50.0]),
        np.array([100.0, 100.0, -50.0]),
        np.array([0.0, 100.0, -50.0]),
        np.array([0.0, 0.0, -50.0]),
    ]


@pytest.fixture
def drone_state_dict():
    """Sample drone state dictionary for monitors."""
    return {
        "altitude": 50.0,
        "speed": 5.0,
        "battery_soc": 0.8,
        "min_obstacle_dist": 15.0,
        "roll_deg": 5.0,
        "pitch_deg": 3.0,
        "yaw_deg": 45.0,
        "min_motor_health": 0.95,
        "gps_quality": 0.9,
        "wind_speed": 3.0,
        "comm_quality": 0.95,
        "threat_level": 0.0,
        "geofence_violated": False,
        "emergency_mode": False,
        "autonomous_mode": True,
        "heartbeat_age_s": 0.5,
        "spoofing_detected": False,
    }


@pytest.fixture
def shared_key():
    """Shared key for crypto tests."""
    return b'\x01' * 32
