import unittest
import sys
from src.simulate import simulate_data
from src.recover import inverse_equations

class TestEZDiffusionModel(unittest.TestCase):
    def test_simulate_data(self):
        """Test that simulate_data() returns valid values."""
        try:
            N = 40
            a, v, t, R_obs, M_obs, V_obs = simulate_data(N)
            
            self.assertTrue(0.5 <= a <= 2, "Boundary separation out of range")
            self.assertTrue(0.5 <= v <= 2, "Drift rate out of range")
            self.assertTrue(0.1 <= t <= 0.5, "Nondecision time out of range")
        except Exception as e:
            print(f"Test failed: {e}")
            sys.exit(1)

    def test_forward_inverse(self):
        """Test that inverse equations recover parameters correctly."""
        try:
            a, v, t = 1.0, 1.0, 0.3
            R, M, V = simulate_data(100)[-3:]
            v_est, a_est, t_est = inverse_equations(R, M, V)

            self.assertAlmostEqual(v, v_est, places=2, msg="Drift rate recovery failed")
            self.assertAlmostEqual(a, a_est, places=2, msg="Boundary separation recovery failed")
            self.assertAlmostEqual(t, t_est, places=2, msg="Nondecision time recovery failed")
        except Exception as e:
            print(f"Test failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    unittest.main()

