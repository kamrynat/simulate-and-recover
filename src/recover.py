#Used ChatGPT throughout to debug
import numpy as np
import sys

def inverse_equations(R_obs, M_obs, V_obs):
    try:
        R_obs = np.clip(R_obs, 1e-6, 1 - 1e-3)


        L = np.log(R_obs / (1 - R_obs))

        V_obs = max(V_obs, 1e-6)

        v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(
            L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs
        )

        v_est = np.clip(v_est, 0.5, 2)

        a_est = L / v_est

        t_est = M_obs - (a_est / (2 * v_est)) * (
            (1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))
        )

        # Debugging print
        print(f"DEBUG: R_obs={R_obs}, M_obs={M_obs}, V_obs={V_obs}, v_est={v_est}, a_est={a_est}, t_est={t_est}")

        return v_est, a_est, t_est
    except Exception as e:
        print(f"Error in inverse_equations: {e}")
        sys.exit(1)
