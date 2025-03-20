#Used ChatGPT throughout to debug code
import numpy as np
import scipy.stats as stats
import sys

A_RANGE = (0.5, 2)
V_RANGE = (0.5, 2)
T_RANGE = (0.1, 0.5)

def forward_equations(a, v, t):
    try:
        y = np.exp(-a * v)
        R_pred = 1 / (y + 1)
        M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((y + 1) ** 2))
        return R_pred, M_pred, V_pred
    except Exception as e:
        print(f"Error in forward_equations: {e}")
        sys.exit(1)

def simulate_data(N):
    try:
        a = np.random.uniform(*A_RANGE)
        v = np.random.uniform(*V_RANGE)
        t = np.random.uniform(*T_RANGE)

        R_pred, M_pred, V_pred = forward_equations(a, v, t)

        R_obs = stats.binom.rvs(N, R_pred) / N
        M_obs = stats.norm.rvs(M_pred, np.sqrt(V_pred / N))
        V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))

        # Debugging print
        print(f"DEBUG: a={a}, v={v}, t={t}, R_pred={R_pred}, M_pred={M_pred}, V_pred={V_pred}, R_obs={R_obs}, M_obs={M_obs}, V_obs={V_obs}")

        return a, v, t, R_obs, M_obs, V_obs
    except Exception as e:
        print(f"Error in simulate_data: {e}")
        sys.exit(1)
