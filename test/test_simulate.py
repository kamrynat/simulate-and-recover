import pytest
from src.simulate import forward_equations, inverse_equations

def test_forward_inverse():
    a, v, t = 1.0, 1.0, 0.3
    R, M, V = forward_equations(a, v, t)
    v_est, a_est, t_est = inverse_equations(R, M, V)
    
    assert abs(v - v_est) < 0.01
    assert abs(a - a_est) < 0.01
    assert abs(t - t_est) < 0.01

if __name__ == "__main__":
    pytest.main()
