# Domain randomization and SE(3) rigid-body plant for sim-to-real transfer.
# Randomizes mass (±40%), motor gain (±25%), inertia (±15%), wind (σ=2 m/s).
# RigidBodyPlant integrates full Euler rotational dynamics with RK4 and SO(3) projection.
#
# Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems

from typing import Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PhysicsParams:
    """Randomized physics parameters for quadrotor"""
    mass: float                    # kg
    inertia: np.ndarray           # 3x3 inertia tensor (kg·m²)
    motor_thrust_coeff: float     # Thrust coefficient C_t
    motor_torque_coeff: float     # Torque coefficient C_q
    drag_coeff: float             # Aerodynamic drag coefficient
    arm_length: float             # Distance from center to motor (m)
    motor_time_constant: float    # Motor response time (s)


@dataclass
class SensorNoiseParams:
    """Sensor noise characteristics"""
    imu_accel_noise: float        # m/s² std dev
    imu_gyro_noise: float         # rad/s std dev
    gps_position_noise: float     # m std dev
    gps_dropout_prob: float       # Probability of GPS dropout
    barometer_noise: float        # m std dev
    magnetometer_noise: float     # rad std dev


@dataclass
class EnvironmentParams:
    """Environmental disturbances"""
    wind_mean: np.ndarray         # Mean wind velocity [vx, vy, vz] (m/s)
    wind_std: float               # Wind turbulence std dev (m/s)
    wind_model: str               # "dryden" or "von_karman"
    air_density: float            # kg/m³
    gravity: float                # m/s²
    ground_effect_height: float   # Height where ground effect starts (m)


class DomainRandomizer:
    """
    Domain randomization for Isaac Sim / PyBullet environments.
    
    Applies massive variation to:
        - Physical parameters (mass, inertia, motor constants)
        - Sensor noise characteristics
        - Environmental conditions (wind, air density)
    """
    
    def __init__(self, seed: int = 42):
        """Initialize domain randomizer with RNG seed"""
        self.rng = np.random.default_rng(seed)
        
        # Nominal parameters (baseline quadrotor)
        self.nominal_mass = 2.0  # kg
        self.nominal_inertia = np.diag([0.0347, 0.0458, 0.0977])  # kg·m²
        self.nominal_thrust_coeff = 1.0
        self.nominal_drag = 0.1
        
    def randomize_physics(self) -> PhysicsParams:
        """
        Randomize physical parameters with aggressive variation.
        
        Returns:
            PhysicsParams with randomized values
        """
        # Mass: ±30% variation
        mass = self.nominal_mass * self.rng.uniform(0.7, 1.3)
        
        # Inertia: ±15% variation per axis
        inertia_scale = self.rng.uniform(0.85, 1.15, size=3)
        inertia = self.nominal_inertia * inertia_scale
        
        # Motor thrust coefficient: ±20% variation
        thrust_coeff = self.nominal_thrust_coeff * self.rng.uniform(0.8, 1.2)
        
        # Torque coefficient: ±20% variation
        torque_coeff = 0.05 * self.rng.uniform(0.8, 1.2)
        
        # Drag coefficient: ±25% variation
        drag_coeff = self.nominal_drag * self.rng.uniform(0.75, 1.25)
        
        # Arm length: ±5% variation (mechanical tolerance)
        arm_length = 0.25 * self.rng.uniform(0.95, 1.05)
        
        # Motor time constant: ±30% variation
        motor_tau = 0.02 * self.rng.uniform(0.7, 1.3)
        
        return PhysicsParams(
            mass=mass,
            inertia=inertia,
            motor_thrust_coeff=thrust_coeff,
            motor_torque_coeff=torque_coeff,
            drag_coeff=drag_coeff,
            arm_length=arm_length,
            motor_time_constant=motor_tau
        )
    
    def randomize_sensors(self) -> SensorNoiseParams:
        """
        Randomize sensor noise characteristics.
        
        Returns:
            SensorNoiseParams with randomized noise levels
        """
        return SensorNoiseParams(
            imu_accel_noise=self.rng.uniform(0.01, 0.1),      # m/s²
            imu_gyro_noise=self.rng.uniform(0.001, 0.01),     # rad/s
            gps_position_noise=self.rng.uniform(1.0, 5.0),    # m
            gps_dropout_prob=self.rng.uniform(0.0, 0.2),      # 0-20% dropout
            barometer_noise=self.rng.uniform(0.1, 1.0),       # m
            magnetometer_noise=self.rng.uniform(0.01, 0.1)    # rad
        )
    
    def randomize_environment(self) -> EnvironmentParams:
        """
        Randomize environmental conditions.
        
        Returns:
            EnvironmentParams with randomized disturbances
        """
        # Wind: mean velocity 0-5 m/s, turbulence 0-2 m/s
        wind_mean = self.rng.uniform(-5.0, 5.0, size=3)
        wind_std = self.rng.uniform(0.0, 2.0)
        
        # Wind model: randomly select Dryden or von Kármán
        wind_model = self.rng.choice(["dryden", "von_karman"])
        
        # Air density: ±10% variation (altitude/temperature effects)
        air_density = 1.225 * self.rng.uniform(0.9, 1.1)  # kg/m³
        
        # Gravity: ±0.5% variation (latitude effects)
        gravity = 9.81 * self.rng.uniform(0.995, 1.005)
        
        # Ground effect height: 0.5-2.0 m
        ground_effect_height = self.rng.uniform(0.5, 2.0)
        
        return EnvironmentParams(
            wind_mean=wind_mean,
            wind_std=wind_std,
            wind_model=wind_model,
            air_density=air_density,
            gravity=gravity,
            ground_effect_height=ground_effect_height
        )
    
    def randomize_all(self) -> Tuple[PhysicsParams, SensorNoiseParams, EnvironmentParams]:
        """
        Randomize all parameters simultaneously.
        
        Returns:
            Tuple of (physics, sensors, environment) parameters
        """
        return (
            self.randomize_physics(),
            self.randomize_sensors(),
            self.randomize_environment()
        )
    
    def apply_motor_failure(self, num_motors: int = 4) -> np.ndarray:
        """
        Simulate motor failure scenario.
        
        Args:
            num_motors: Total number of motors
            
        Returns:
            Motor health array [0.0-1.0] per motor (1.0 = healthy)
        """
        motor_health = np.ones(num_motors)
        
        # 10% chance of partial motor failure
        if self.rng.random() < 0.1:
            failed_motor = self.rng.integers(0, num_motors)
            motor_health[failed_motor] = self.rng.uniform(0.3, 0.7)
        
        return motor_health


def test_domain_randomization():
    """Test domain randomization with statistics"""
    randomizer = DomainRandomizer(seed=42)
    
    # Generate 1000 samples
    masses = []
    for _ in range(1000):
        params = randomizer.randomize_physics()
        masses.append(params.mass)
    
    masses = np.array(masses)
    print(f"Mass distribution:")
    print(f"  Mean: {masses.mean():.3f} kg")
    print(f"  Std:  {masses.std():.3f} kg")
    print(f"  Min:  {masses.min():.3f} kg")
    print(f"  Max:  {masses.max():.3f} kg")
    
    # Test full randomization
    physics, sensors, env = randomizer.randomize_all()
    print(f"\nSample randomization:")
    print(f"  Mass: {physics.mass:.3f} kg")
    print(f"  Thrust coeff: {physics.motor_thrust_coeff:.3f}")
    print(f"  GPS noise: {sensors.gps_position_noise:.3f} m")
    print(f"  Wind mean: {env.wind_mean}")
    print(f"  Wind model: {env.wind_model}")


# ─── SE(3) Rigid-Body Dynamics Plant ─────────────────────────────────────────
# Replaces the kinematic F=ma point-mass assumption with full Euler rigid-body
# rotational dynamics.  Inputs: collective thrust T (N) + body torques tau (N·m).
# State: position p (m), velocity v (m/s), rotation matrix R ∈ SO(3), body
# angular velocity omega (rad/s).
#
# Translational dynamics (world frame):
#   p_dot = v
#   v_dot = (T/m) * R @ e3 - g*e3 + f_wind/m
#
# Rotational dynamics (body frame — Euler's equation):
#   R_dot   = R @ hat(omega)
#   J*omega_dot = tau - omega × (J*omega)
#
# where hat(·) is the skew-symmetric operator and J = diag(Ixx, Iyy, Izz).
#
# References:
#   Lee et al. (2010). Geometric Tracking Control of a Quadrotor UAV on SE(3). CDC.
#   Mahony et al. (2012). Multirotor Aerial Vehicles. IEEE RAM.

@dataclass
class SE3State:
    """Full rigid-body state on SE(3)."""
    p:     np.ndarray  # position (3,)  world frame
    v:     np.ndarray  # velocity (3,)  world frame
    R:     np.ndarray  # rotation (3,3) body→world
    omega: np.ndarray  # angular velocity (3,) body frame


def _hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector (hat map)."""
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


class RigidBodyPlant:
    """
    SE(3) rigid-body quadrotor dynamics plant.

    Inputs at each step:
        T   : collective thrust (N)
        tau : body torques [tau_x, tau_y, tau_z] (N·m)

    Integrates with 4th-order Runge-Kutta.  R is re-orthogonalised every step
    via SVD to prevent numerical drift off SO(3).
    """

    # Nominal inertia from architecture spec
    _J_NOM = np.diag([0.0347, 0.0458, 0.0977])  # kg·m²
    _E3    = np.array([0.0, 0.0, 1.0])
    _G     = 9.81  # m/s²

    def __init__(self, mass: float = 2.0, J: np.ndarray | None = None,
                 wind_sigma: float = 0.05, drag_k: float = 0.1):
        """
        Args:
            mass       : vehicle mass (kg)
            J          : 3×3 inertia tensor; defaults to nominal spec values
            wind_sigma : wind disturbance std dev (m/s) — spec: 0.05
            drag_k     : translational drag coefficient (m⁻¹) — spec: 0.1
        """
        self.m    = mass
        self.J    = J if J is not None else self._J_NOM.copy()
        self.Jinv = np.linalg.inv(self.J)
        self.wind_sigma = wind_sigma
        self.drag_k     = drag_k
        self._rng = np.random.default_rng(0)

    def reset(self, p0: np.ndarray | None = None) -> SE3State:
        """Return initial state with identity attitude."""
        return SE3State(
            p     = p0.copy() if p0 is not None else np.array([0.0, 0.0, 2.0]),
            v     = np.zeros(3),
            R     = np.eye(3),
            omega = np.zeros(3),
        )

    def _derivatives(self, s: SE3State, T: float, tau: np.ndarray,
                     wind: np.ndarray) -> SE3State:
        """Continuous-time derivatives — returns dState/dt as SE3State."""
        # Translational
        thrust_world = T / self.m * (s.R @ self._E3)
        drag         = -self.drag_k * s.v
        p_dot = s.v
        v_dot = thrust_world - self._G * self._E3 + drag + wind / self.m

        # Rotational — Euler's equation in body frame
        R_dot     = s.R @ _hat(s.omega)
        Jomega    = self.J @ s.omega
        omega_dot = self.Jinv @ (tau - np.cross(s.omega, Jomega))

        return SE3State(p=p_dot, v=v_dot, R=R_dot, omega=omega_dot)

    @staticmethod
    def _project_SO3(R: np.ndarray) -> np.ndarray:
        """Project R back onto SO(3) via SVD to prevent numerical drift."""
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

    def step(self, state: SE3State, T: float, tau: np.ndarray,
             dt: float = 0.1) -> SE3State:
        """
        Advance state by dt seconds using RK4.

        Args:
            state : current SE3State
            T     : collective thrust (N)
            tau   : body torques (N·m), shape (3,)
            dt    : timestep (s); default 0.1 s = 10 Hz loop

        Returns:
            New SE3State after dt seconds.
        """
        wind = self._rng.normal(0.0, self.wind_sigma, 3)

        def _add(s: SE3State, ds: SE3State, h: float) -> SE3State:
            return SE3State(
                p     = s.p     + h * ds.p,
                v     = s.v     + h * ds.v,
                R     = s.R     + h * ds.R,
                omega = s.omega + h * ds.omega,
            )

        k1 = self._derivatives(state,              T, tau, wind)
        k2 = self._derivatives(_add(state, k1, dt/2), T, tau, wind)
        k3 = self._derivatives(_add(state, k2, dt/2), T, tau, wind)
        k4 = self._derivatives(_add(state, k3, dt),   T, tau, wind)

        new = SE3State(
            p     = state.p     + (dt/6)*(k1.p     + 2*k2.p     + 2*k3.p     + k4.p),
            v     = state.v     + (dt/6)*(k1.v     + 2*k2.v     + 2*k3.v     + k4.v),
            R     = state.R     + (dt/6)*(k1.R     + 2*k2.R     + 2*k3.R     + k4.R),
            omega = state.omega + (dt/6)*(k1.omega + 2*k2.omega + 2*k3.omega + k4.omega),
        )
        new.R = self._project_SO3(new.R)
        # Write ground-truth velocity to /dev/shm/aisp_gt_state so the EKF
        # VIO injection has a real air-gap source (vx, vy from physics plant).
        try:
            import mmap as _mmap, struct as _struct, os as _os
            _fmt  = "=dd"
            _size = _struct.calcsize(_fmt)
            _fd   = _os.open("/dev/shm/aisp_gt_state", _os.O_CREAT | _os.O_RDWR, 0o666)
            _os.ftruncate(_fd, _size)
            _shm  = _mmap.mmap(_fd, _size, _mmap.MAP_SHARED, _mmap.PROT_WRITE)
            _os.close(_fd)
            _shm.seek(0)
            _shm.write(_struct.pack(_fmt, float(new.v[0]), float(new.v[1])))
            _shm.close()
        except OSError:
            pass
        return new


def test_rigid_body_plant():
    """Smoke-test: hover thrust should keep altitude near-constant for 5 s."""
    plant = RigidBodyPlant()
    state = plant.reset(p0=np.array([0.0, 0.0, 2.0]))

    T_hover = plant.m * plant._G          # exact hover thrust
    tau     = np.zeros(3)                 # no torques → no rotation
    dt      = 0.1
    steps   = 50                          # 5 seconds at 10 Hz

    for _ in range(steps):
        state = plant.step(state, T_hover, tau, dt)

    # With wind σ=0.05 and drag, altitude should stay within ±0.5 m of 2.0 m
    assert abs(state.p[2] - 2.0) < 0.5, (
        f"SE(3) plant hover test FAILED: pz={state.p[2]:.3f} m"
    )
    # R must remain in SO(3): det(R) ≈ 1
    assert abs(np.linalg.det(state.R) - 1.0) < 1e-6, "R drifted off SO(3)"
    print(f"SE(3) rigid-body plant: PASS  (pz={state.p[2]:.3f} m after 5 s hover)")
    print(f"  det(R) = {np.linalg.det(state.R):.8f}  (should be 1.0)")
    print(f"  omega  = {state.omega}  (should be ~0)")


if __name__ == "__main__":
    test_domain_randomization()
    print()
    test_rigid_body_plant()
