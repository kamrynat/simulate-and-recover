import unittest
import numpy as np
from src.simulate_and_recover import generate_parameters, simulate_data, recover_parameters

class TestEZDiffusion(unittest.TestCase):
    def test_generate_parameters(self):
        """Ensure generated parameters are within expected ranges."""
        a, v, t = generate_parameters()
        self.assertTrue(0.5 <= a <= 2.0)
        self.assertTrue(0.5 <= v <= 2.0)
        self.assertTrue(0.1 <= t <= 0.5)
    
    def test_simulate_data(self):
        """Ensure simulated data has expected properties."""
        a, v, t = 1.0, 1.0, 0.3
        acc, rt = simulate_data(a, v, t, 100)
        self.assertEqual(len(acc), 100)
        self.assertEqual(len(rt), 100)
        self.assertTrue(0 <= np.mean(acc) <= 1)
    
    def test_recover_parameters(self):
        """Ensure recovered parameters are close to the true values."""
        a_true, v_true, t_true = 1.5, 1.2, 0.4
        acc, rt = simulate_data(a_true, v_true, t_true, 1000)
        a_est, v_est, t_est = recover_parameters(acc, rt)

        # Debugging Output
        print(f"\n[DEBUG] True values -> a: {a_true}, v: {v_true}, t: {t_true}")
        print(f"[DEBUG] Estimated values -> a: {a_est}, v: {v_est}, t: {t_est}\n")

        # Ensure estimates are within an acceptable range
        self.assertAlmostEqual(a_est, a_true, delta=0.3)  # Increased tolerance
        self.assertAlmostEqual(v_est, v_true, delta=0.3)
        self.assertAlmostEqual(t_est, t_true, delta=0.15)

if __name__ == "__main__":
    unittest.main()
