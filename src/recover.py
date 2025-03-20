#Used ChatGPT throughout to debug
import numpy as np
import scipy.stats as stats
import sys

def inverse_equations(R_obs, M_obs, V_obs):
    try:
        R_obs = np.clip(R_obs, 1e-6, 1 - 1e-6)

        L = np.log(R_obs / (1 - R_obs))
        v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs ** 2 * L - R_obs * L + R_obs - 0.5) / V_obs)
        a_est = L / v_est
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))

        return v_est, a_est, t_est
    except Exception as e:
        print(f"Error in inverse_equations: {e}")
        sys.exit(1)
