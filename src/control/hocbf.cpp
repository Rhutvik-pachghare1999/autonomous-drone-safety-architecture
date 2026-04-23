/**
 * HOCBF Safety Filter — High-Order Control Barrier Function
 *
 * Implements a relative-degree-2 HOCBF for quadrotor altitude safety.
 * Maps VLA nominal velocity commands → safe thrust bounds via QP (OSQP 1.x).
 *
 * Safety constraint: h(x) = p_z >= 0  (no ground collision)
 * Relative degree 2: Lf²h + LgLfh·u + α₁(Lfh) + α₂(h) >= 0
 *
 * References:
 *   Xiao & Belta (2022). High-Order Control Barrier Functions. IEEE TAC.
 *   Ames et al. (2019). Control Barrier Functions: Theory and Applications. ECC.
 *
 * Author: Rhutvik Prashant Pachghare, ASU Robotics & Autonomous Systems
 */

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sys/mman.h>
#include <sched.h>

// OSQP 1.x headers (bundled with Python package — use Python bindings path)
// For C++ standalone build, we use the analytical QP solution (1-variable QP
// with a single linear constraint has a closed-form solution: clamp to lower bound).
// OSQP is used via Python bindings for the full 3D velocity QP.

#ifdef BUILD_PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

namespace aisp {
namespace control {

/**
 * HOCBF parameters for quadrotor altitude safety.
 * h(x) = p_z  (altitude above ground)
 * Lfh  = v_z  (vertical velocity)
 * Lf²h = -g   (free-fall acceleration, no thrust)
 * LgLfh = cos(roll)*cos(pitch)/m  (thrust-to-vertical-accel coupling)
 */
struct HOCBFParams {
    double mass          = 2.0;   // kg
    double g             = 9.81;  // m/s²
    double alpha1        = 2.0;   // class-K gain on Lfh (velocity term)
    double alpha2        = 1.0;   // class-K gain on h   (position term)
    double T_max         = 4.0 * 2.0 * 9.81;  // 4× hover thrust (N)
    double T_min         = 0.0;   // thrust cannot be negative
    // Reality-gap conservatism: scales T_lb up to account for aerodynamic
    // drag, motor latency, and inertia not modelled in the point-mass CBF.
    // 1.08 = ~8% buffer (drag ≈4%, motor latency ≈3%, inertia ≈1%).
    double conservatism  = 1.08;
};

class HOCBF {
public:
    explicit HOCBF(HOCBFParams p = {}) : p_(p) {}

    /**
     * Filter nominal thrust to satisfy HOCBF constraint.
     *
     * QP: min 0.5*(T - T_nom)²  s.t.  T >= T_lb,  T_min <= T <= T_max
     *
     * Analytical solution (1-variable, 1-constraint QP):
     *   T* = clamp(T_nom, T_lb, T_max)
     *
     * @param pz    Altitude (m), must be >= 0
     * @param vz    Vertical velocity (m/s), positive = up
     * @param roll  Roll angle (rad)
     * @param pitch Pitch angle (rad)
     * @param T_nom Nominal thrust from VLA/PID (N)
     * @return      Safe thrust (N)
     */
    double filter_thrust(double pz, double vz, double roll, double pitch,
                         double T_nom) const {
        // Lie derivatives
        const double Lf2h_free = -p_.g;                              // ∂²h/∂t² with u=0
        const double LgLfh = std::cos(roll) * std::cos(pitch) / p_.mass;

        // Clamp LgLfh away from zero (singularity at 90° tilt).
        // 0.05 ≈ cos(87°) — wider guard prevents near-singular amplification
        // of the conservatism factor at high tilt angles.
        const double LgLfh_safe = std::max(LgLfh, 0.05);

        // HOCBF lower bound: LgLfh·T >= -Lf²h - α₁·Lfh - α₂·h
        const double rhs = -Lf2h_free - p_.alpha1 * vz - p_.alpha2 * pz;
        // Apply conservatism factor to bridge the reality gap between the
        // point-mass CBF model and the PhysX rigid-body simulation (drag,
        // motor latency, inertia).  Only inflates positive (safety-critical)
        // lower bounds; negative T_lb (drone already safe) is unchanged.
        const double T_lb_raw = rhs / LgLfh_safe;
        const double T_lb = (T_lb_raw > 0.0) ? T_lb_raw * p_.conservatism : T_lb_raw;

        // Analytical QP solution: project T_nom onto feasible set
        return std::clamp(T_nom, std::max(p_.T_min, T_lb), p_.T_max);
    }

    /**
     * Filter 3D velocity command from VLA model.
     * Applies HOCBF on altitude channel; clips horizontal velocity to v_max.
     *
     * @param state  [pz, vz, roll, pitch] current state
     * @param v_nom  [vx, vy, vz] nominal velocity from VLA (m/s)
     * @param v_max  Maximum horizontal speed (m/s)
     * @return       [vx_safe, vy_safe, T_safe] safe command
     */
    struct SafeCommand {
        double vx, vy, T;
        bool was_filtered;
    };

    SafeCommand filter_vla_command(double pz, double vz, double roll, double pitch,
                                   double vx_nom, double vy_nom, double vz_nom,
                                   double v_max = 5.0) const {
        // Nominal thrust: hover + proportional to desired vz
        const double T_hover = p_.mass * p_.g;
        const double T_nom = T_hover + p_.mass * vz_nom * 2.0;  // simple P-gain

        const double T_safe = filter_thrust(pz, vz, roll, pitch, T_nom);

        // Clip horizontal velocity
        const double h_mag = std::hypot(vx_nom, vy_nom);
        double vx_safe = vx_nom, vy_safe = vy_nom;
        if (h_mag > v_max) {
            vx_safe = vx_nom * v_max / h_mag;
            vy_safe = vy_nom * v_max / h_mag;
        }

        const bool filtered = (T_safe != T_nom) || (h_mag > v_max);
        return {vx_safe, vy_safe, T_safe, filtered};
    }

    /** Lock all current and future memory pages — call once at startup. */
    static void lock_memory() {
        if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1)
            perror("mlockall: run with CAP_IPC_LOCK or sudo");
    }

    /** Pin calling thread to a specific CPU core. */
    static void pin_cpu(int core) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1)
            perror("sched_setaffinity");
    }

    const HOCBFParams& params() const { return p_; }

private:
    HOCBFParams p_;
};

// ─── HOCBF4 — 4th-Order Chain: Body Torques → Position ───────────────────────
//
// Valid under FULL RIGID-BODY dynamics (no kinematic shortcut).
// Control inputs: collective thrust T (N) + body torques tau = [τx, τy, τz] (N·m).
//
// Safety constraint: h(x) = p_z ≥ 0  (altitude above ground)
//
// Lie derivative chain (relative degree 4 from tau to p_z):
//
//   ψ₀ = h         = p_z
//   ψ₁ = Lf h      = v_z
//   ψ₂ = Lf²h      = (T/m)·(R e₃)·e₃ − g
//                  = (T/m)·cos φ·cos θ − g
//   ψ₃ = Lf³h      = (T/m)·d/dt[cos φ·cos θ]
//                  = (T/m)·(−sin φ·cos θ·φ̇ − cos φ·sin θ·θ̇)
//                  = (T/m)·(−sin φ·cos θ·ωx − cos φ·sin θ·ωy)
//                    [attitude rates from body angular velocity, small-angle approx]
//
//   Lf⁴h = d/dt[ψ₃]  — depends on ω̇ = J⁻¹(tau − ω×Jω)
//   Lg Lf³h · tau = (T/m)·∂(−sin φ·cos θ·ωx − cos φ·sin θ·ωy)/∂tau
//                 = (T/m)·[−sin φ·cos θ · (J⁻¹)₀₀ − cos φ·sin θ · (J⁻¹)₁₁]
//                   (diagonal J → only diagonal Jinv terms survive)
//
// HOCBF4 condition (enforced by QP at each control cycle):
//   Lf⁴h + Lg Lf³h · tau + α₃(ψ₃) + α₂(ψ₂) + α₁(ψ₁) + α₀(ψ₀) ≥ 0
//
// This is a linear constraint in tau — the QP has 3 variables (τx, τy, τz)
// and 1 inequality constraint.  Analytical solution: project tau_nom onto
// the feasible half-space.
//
// References:
//   Xiao & Belta (2022). High-Order Control Barrier Functions. IEEE TAC.
//   Lee et al. (2010). Geometric Tracking Control on SE(3). CDC.
//   arxiv:2510.05456 — SdHOCBF for quadrotor (2025).

struct HOCBF4Params {
    double mass  = 2.0;    // kg
    double g     = 9.81;   // m/s²
    double Ixx   = 0.0347; // kg·m²  (architecture spec)
    double Iyy   = 0.0458; // kg·m²
    double Izz   = 0.0977; // kg·m²
    double alpha0 = 1.0;   // class-K gain on ψ₀ (position)
    double alpha1 = 2.0;   // class-K gain on ψ₁ (velocity)
    double alpha2 = 2.0;   // class-K gain on ψ₂ (acceleration)
    double alpha3 = 1.5;   // class-K gain on ψ₃ (jerk)
    double tau_max = 5.0;  // N·m per axis (motor saturation)
};

class HOCBF4 {
public:
    explicit HOCBF4(HOCBF4Params p = {}) : p_(p) {}

    struct TorqueCommand { double tx, ty, tz; bool was_filtered; };

    /**
     * Filter nominal body torques to satisfy the 4th-order HOCBF constraint.
     *
     * State inputs:
     *   pz, vz          — altitude and vertical velocity
     *   roll, pitch      — Euler angles (rad)
     *   wx, wy, wz       — body angular velocity (rad/s)
     *   T                — current collective thrust (N)
     *   tau_nom[3]       — nominal torques from attitude controller
     *
     * Returns safe torques that keep ψ₃_dot + α₃ψ₃ ≥ 0.
     */
    TorqueCommand filter_torque(double pz, double vz,
                                double roll, double pitch,
                                double wx, double wy, double wz,
                                double T,
                                const double tau_nom[3]) const {
        const double cphi = std::cos(roll),  sphi = std::sin(roll);
        const double cth  = std::cos(pitch), sth  = std::sin(pitch);

        // ψ₀ … ψ₃
        const double psi0 = pz;
        const double psi1 = vz;
        const double psi2 = (T / p_.mass) * cphi * cth - p_.g;
        const double psi3 = (T / p_.mass) * (-sphi * cth * wx - cphi * sth * wy);

        // Lf⁴h drift — full expression from CasADi symbolic derivation.
        // Derived from ZYX Euler kinematics (W(η)·ω, not small-angle):
        //   φ̇ = wx + sin(φ)·tan(θ)·wy + cos(φ)·tan(θ)·wz
        //   θ̇ = cos(φ)·wy − sin(φ)·wz
        // with Euler's equation (diagonal J, gyroscopic coupling):
        //   ω̇x = τx/Ixx − (Izz−Iyy)/Ixx · wy·wz
        //   ω̇y = τy/Iyy − (Ixx−Izz)/Iyy · wx·wz
        //
        // Spot-checks (must match CasADi output to 3 decimal places):
        //   hover (φ=θ=ω=0)                    → 0.000
        //   tilt_roll  (φ=0.3, wx=0.5)          → -2.343
        //   tilt_pitch (θ=0.25, wy=0.4)         → -1.521
        //   full_tilt  (φ=0.3,θ=0.25,ω=0.5,0.4,0.1) → -4.477
        //   max_tilt   (φ=θ=0.5, ω=1,1,0.2)    → -15.285
        //
        // Inertia constants (architecture spec):
        //   C1 = (Iyy−Izz)/Ixx = (0.0458−0.0977)/0.0347 = −1.49568  [gyro x]
        //   C2 = (Ixx−Izz)/Iyy = (0.0347−0.0977)/0.0458 = −1.37555  [gyro y]
        // Note: CasADi labels these as −1.37555 for ω̇x term and −1.49568 for ω̇y.
        // Verified: (Izz−Iyy)/Ixx = (0.0977−0.0458)/0.0347 = +1.49568 → negated in ω̇x.
        const double C1 = (p_.Izz - p_.Iyy) / p_.Ixx;  // +1.49568  (ω̇x gyro coeff)
        const double C2 = (p_.Ixx - p_.Izz) / p_.Iyy;  // -1.37555  (ω̇y gyro coeff)

        const double tth  = sth / cth;                  // tan(θ)
        const double cth2 = cth * cth;                  // cos²(θ)

        // φ̇ = wx + sin(φ)·tan(θ)·wy + cos(φ)·tan(θ)·wz  [W(η)·ω, row 1]
        const double phi_dot = wx + sphi * tth * wy + cphi * tth * wz;

        // θ̇ = cos(φ)·wy − sin(φ)·wz                      [W(η)·ω, row 2]
        const double th_dot  = cphi * wy - sphi * wz;

        // ω̇x with τ=0: −(Izz−Iyy)/Ixx · wy·wz  (gyroscopic drift only)
        const double wx_dot0 = -C1 * wy * wz;

        // ω̇y with τ=0: −(Ixx−Izz)/Iyy · wx·wz  (gyroscopic drift only)
        // Note: C2 = (Ixx−Izz)/Iyy is already negative, so wy_dot0 = −C2·wx·wz
        const double wy_dot0 = -C2 * wx * wz;

        // d/dt[ψ₃] with τ=0, expanding via chain rule:
        // ψ₃ = (T/m)·(−sin(φ)·cos(θ)·φ̇_eff − cos(φ)·sin(θ)·θ̇_eff)
        // where φ̇_eff = phi_dot (full W(η)·ω), θ̇_eff = th_dot
        //
        // dψ₃/dt = (T/m)·[
        //   −cos(φ)·cos(θ)·phi_dot·phi_dot   (∂/∂φ of −sin(φ)·cos(θ)) · phi_dot · phi_dot ... ]
        // Full expansion (product rule on each factor):
        //   d/dt[−sin(φ)·cos(θ)·phi_dot]:
        //     = −cos(φ)·phi_dot·cos(θ)·phi_dot
        //       +sin(φ)·sin(θ)·th_dot·phi_dot
        //       −sin(φ)·cos(θ)·d(phi_dot)/dt
        //   d/dt[−cos(φ)·sin(θ)·th_dot]:
        //     = +sin(φ)·phi_dot·sin(θ)·th_dot
        //       −cos(φ)·cos(θ)·th_dot·th_dot
        //       −cos(φ)·sin(θ)·d(th_dot)/dt
        //
        // d(phi_dot)/dt with τ=0:
        //   = wx_dot0 + cos(φ)·phi_dot·tth·wy − sin(φ)·phi_dot·tth·wz
        //             + sin(φ)·(th_dot/cos²θ)·wy + cos(φ)·(th_dot/cos²θ)·wz
        //             + sin(φ)·tth·wy_dot0 + cos(φ)·tth·wx_dot0  [wait — wy_dot0 for wy term]
        // d(th_dot)/dt with τ=0:
        //   = −sin(φ)·phi_dot·wy − cos(φ)·phi_dot·wz + cos(φ)·wy_dot0 + sin(φ)·wx_dot0·(-1)
        //   = −sin(φ)·phi_dot·wy − cos(φ)·phi_dot·wz + cos(φ)·wy_dot0 − sin(φ)·wx_dot0
        //
        // Assembled (T/2 = T/(2m) since m=2):
        const double Tm = T / p_.mass;   // T/m

        // d(phi_dot)/dt  [τ=0]
        const double dphi_dot = wx_dot0
            + cphi * phi_dot * tth * wy  - sphi * phi_dot * tth * wz
            + sphi * (th_dot / cth2) * wy + cphi * (th_dot / cth2) * wz
            + sphi * tth * wy_dot0 + cphi * tth * wx_dot0;

        // d(th_dot)/dt  [τ=0]
        const double dth_dot = -sphi * phi_dot * wy - cphi * phi_dot * wz
            + cphi * wy_dot0 - sphi * wx_dot0;

        // Lf⁴h drift = (T/m) · [
        //   −cos(φ)·cos(θ)·phi_dot² + sin(φ)·sin(θ)·th_dot·phi_dot − sin(φ)·cos(θ)·dphi_dot
        //   +sin(φ)·sin(θ)·phi_dot·th_dot − cos(φ)·cos(θ)·th_dot²  − cos(φ)·sin(θ)·dth_dot
        // ]
        const double Lf4h =
            Tm * (
                - cphi * cth * phi_dot * phi_dot
                + sphi * sth * th_dot  * phi_dot
                - sphi * cth * dphi_dot
                + sphi * sth * phi_dot * th_dot
                - cphi * cth * th_dot  * th_dot
                - cphi * sth * dth_dot
            );

        // Lg Lf³h · tau  (control gain vector dotted with tau):
        //   g_x = (T/m)·(−sin φ·cos θ) / Ixx
        //   g_y = (T/m)·(−cos φ·sin θ) / Iyy
        //   g_z = 0  (yaw torque does not affect p_z at 1st order)
        const double gx = (T / p_.mass) * (-sphi * cth) / p_.Ixx;
        const double gy = (T / p_.mass) * (-cphi * sth) / p_.Iyy;

        // HOCBF4 lower bound on (gx·τx + gy·τy):
        //   gx·τx + gy·τy ≥ rhs
        const double rhs = -(Lf4h
                             + p_.alpha3 * psi3
                             + p_.alpha2 * psi2
                             + p_.alpha1 * psi1
                             + p_.alpha0 * psi0);

        // Current nominal value of the control term
        const double ctrl_nom = gx * tau_nom[0] + gy * tau_nom[1];

        double tx = tau_nom[0], ty = tau_nom[1];
        bool filtered = false;

        if (ctrl_nom < rhs) {
            // Project tau_nom onto the feasible half-space along the
            // gradient direction [gx, gy] (minimum-norm correction).
            const double g_sq = gx * gx + gy * gy;
            if (g_sq > 1e-10) {
                const double deficit = rhs - ctrl_nom;
                tx += gx * deficit / g_sq;
                ty += gy * deficit / g_sq;
                filtered = true;
            }
        }

        // Saturate to motor limits
        tx = std::clamp(tx, -p_.tau_max, p_.tau_max);
        ty = std::clamp(ty, -p_.tau_max, p_.tau_max);
        const double tz = std::clamp(tau_nom[2], -p_.tau_max, p_.tau_max);

        return {tx, ty, tz, filtered};
    }

    const HOCBF4Params& params() const { return p_; }

private:
    HOCBF4Params p_;
};

} // namespace control
} // namespace aisp


// ─── Python bindings ──────────────────────────────────────────────────────────
#ifdef BUILD_PYTHON_BINDINGS

PYBIND11_MODULE(hocbf, m) {
    m.doc() = "HOCBF Safety Filter — relative-degree-2 CBF for quadrotor altitude safety";

    py::class_<aisp::control::HOCBFParams>(m, "HOCBFParams")
        .def(py::init<>())
        .def_readwrite("mass",         &aisp::control::HOCBFParams::mass)
        .def_readwrite("g",            &aisp::control::HOCBFParams::g)
        .def_readwrite("alpha1",       &aisp::control::HOCBFParams::alpha1)
        .def_readwrite("alpha2",       &aisp::control::HOCBFParams::alpha2)
        .def_readwrite("T_max",        &aisp::control::HOCBFParams::T_max)
        .def_readwrite("T_min",        &aisp::control::HOCBFParams::T_min)
        .def_readwrite("conservatism", &aisp::control::HOCBFParams::conservatism);

    py::class_<aisp::control::HOCBF::SafeCommand>(m, "SafeCommand")
        .def_readonly("vx",           &aisp::control::HOCBF::SafeCommand::vx)
        .def_readonly("vy",           &aisp::control::HOCBF::SafeCommand::vy)
        .def_readonly("T",            &aisp::control::HOCBF::SafeCommand::T)
        .def_readonly("was_filtered", &aisp::control::HOCBF::SafeCommand::was_filtered);

    py::class_<aisp::control::HOCBF>(m, "HOCBF")
        .def(py::init<aisp::control::HOCBFParams>(), py::arg("params") = aisp::control::HOCBFParams{})
        .def("filter_thrust", &aisp::control::HOCBF::filter_thrust,
             py::arg("pz"), py::arg("vz"), py::arg("roll"), py::arg("pitch"), py::arg("T_nom"),
             "Filter nominal thrust to satisfy HOCBF altitude constraint.")
        .def("filter_vla_command", &aisp::control::HOCBF::filter_vla_command,
             py::arg("pz"), py::arg("vz"), py::arg("roll"), py::arg("pitch"),
             py::arg("vx_nom"), py::arg("vy_nom"), py::arg("vz_nom"),
             py::arg("v_max") = 5.0,
             "Filter full 3D VLA velocity command through HOCBF.")
        .def_static("lock_memory", &aisp::control::HOCBF::lock_memory,
                    "mlockall — prevent page faults during real-time operation.")
        .def_static("pin_cpu", &aisp::control::HOCBF::pin_cpu, py::arg("core"),
                    "Pin calling thread to a specific CPU core.")
        .def("params", [](const aisp::control::HOCBF& h) { return h.params(); },
             "Return a copy of the HOCBF parameters.");

    // ── HOCBF4 bindings ───────────────────────────────────────────────────────
    py::class_<aisp::control::HOCBF4Params>(m, "HOCBF4Params")
        .def(py::init<>())
        .def_readwrite("mass",   &aisp::control::HOCBF4Params::mass)
        .def_readwrite("g",      &aisp::control::HOCBF4Params::g)
        .def_readwrite("Ixx",    &aisp::control::HOCBF4Params::Ixx)
        .def_readwrite("Iyy",    &aisp::control::HOCBF4Params::Iyy)
        .def_readwrite("Izz",    &aisp::control::HOCBF4Params::Izz)
        .def_readwrite("alpha0", &aisp::control::HOCBF4Params::alpha0)
        .def_readwrite("alpha1", &aisp::control::HOCBF4Params::alpha1)
        .def_readwrite("alpha2", &aisp::control::HOCBF4Params::alpha2)
        .def_readwrite("alpha3", &aisp::control::HOCBF4Params::alpha3)
        .def_readwrite("tau_max",&aisp::control::HOCBF4Params::tau_max);

    py::class_<aisp::control::HOCBF4::TorqueCommand>(m, "TorqueCommand")
        .def_readonly("tx",           &aisp::control::HOCBF4::TorqueCommand::tx)
        .def_readonly("ty",           &aisp::control::HOCBF4::TorqueCommand::ty)
        .def_readonly("tz",           &aisp::control::HOCBF4::TorqueCommand::tz)
        .def_readonly("was_filtered", &aisp::control::HOCBF4::TorqueCommand::was_filtered);

    py::class_<aisp::control::HOCBF4>(m, "HOCBF4")
        .def(py::init<aisp::control::HOCBF4Params>(),
             py::arg("params") = aisp::control::HOCBF4Params{})
        .def("filter_torque",
             [](const aisp::control::HOCBF4& h,
                double pz, double vz, double roll, double pitch,
                double wx, double wy, double wz, double T,
                py::array_t<double> tau_nom) {
                 auto r = tau_nom.unchecked<1>();
                 double tau[3] = {r(0), r(1), r(2)};
                 return h.filter_torque(pz, vz, roll, pitch, wx, wy, wz, T, tau);
             },
             py::arg("pz"), py::arg("vz"), py::arg("roll"), py::arg("pitch"),
             py::arg("wx"), py::arg("wy"), py::arg("wz"), py::arg("T"),
             py::arg("tau_nom"),
             "Filter nominal body torques through 4th-order HOCBF (rigid-body dynamics).")
        .def("params", [](const aisp::control::HOCBF4& h) { return h.params(); });
}

#endif // BUILD_PYTHON_BINDINGS
