import unittest

import numpy as np

from app.core.utils import cumulative_trapezoidal, integrate_velocity_from_imu_trapezoidal


class TestImuIntegrationUtils(unittest.TestCase):
    def test_cumulative_trapezoidal_linear_signal(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        v = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        integrated = cumulative_trapezoidal(t, v)
        self.assertTrue(np.allclose(integrated, np.array([0.0, 0.5, 2.0, 4.5], dtype=float)))

    def test_integrate_velocity_from_imu_trapezoidal_stationary(self) -> None:
        t = np.array([0.0, 1.0, 2.0], dtype=float)
        # Stationary body frame with gravity in +Z and zero gyro.
        acc = np.array([[0.0, 0.0, 9.80665], [0.0, 0.0, 9.80665], [0.0, 0.0, 9.80665]], dtype=float)
        gyro = np.zeros((3, 3), dtype=float)

        out = integrate_velocity_from_imu_trapezoidal(t, acc, gyro)

        self.assertTrue(np.allclose(out["vel_earth_x"], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(out["vel_earth_y"], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(out["vel_earth_z"], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
