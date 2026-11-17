"""Regression guard: 1D sweep must not inherit Crazyflie default thrust bounds."""

from experiments.exp_adversarial_1000 import MASS, GRAVITY, simulate_trial


def test_explicit_point_mass_sweep_configuration():
    """A 2 kg model needs sufficient thrust; a 0.60 N default cannot hover."""
    for command in (-0.1, -3.5, -100.0):
        trial = simulate_trial(command)
        assert trial["survived"], (command, trial)
        assert 0.0 <= trial["T_safe"] <= 4.0 * MASS * GRAVITY
    assert simulate_trial(-100.0)["hocbf_triggered"]
