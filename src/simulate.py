#Used ChatGPT to debug code
import numpy as np
import pandas as pd

# Define parameter ranges
A_RANGE = (0.5, 2)  # Boundary separation α
V_RANGE = (0.5, 2)  # Drift rate ν
T_RANGE = (0.1, 0.5)  # Nondecision time τ

# Forward equations to generate predicted summary statistics
def forward_equations(a, v, t):
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v ** 3)) * ((1 - 2 * a * v * y - y ** 2) / ((y + 1) ** 2))
    return R_pred, M_pred, V_pred

# Inverse equations to recover parameters
def inverse_equations(R_obs, M_obs, V_obs):
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs ** 2 * L - R_obs * L + R_obs - 0.5) / V_obs)
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    return v_est, a_est, t_est

# Simulate and recover for different sample sizes
def simulate_and_recover(N, iterations=1000):
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        # Generate random parameters
        a_true = np.random.uniform(*A_RANGE)
        v_true = np.random.uniform(*V_RANGE)
        t_true = np.random.uniform(*T_RANGE)

        # Compute predicted values
        R_pred, M_pred, V_pred = forward_equations(a_true, v_true, t_true)

        # Simulate observed values
        R_obs = np.random.binomial(N, R_pred) / N
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
        V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))

        # Recover parameters
        v_est, a_est, t_est = inverse_equations(R_obs, M_obs, V_obs)

        # Compute bias and squared error
        bias = (v_true - v_est, a_true - a_est, t_true - t_est)
        squared_error = (bias[0] ** 2, bias[1] ** 2, bias[2] ** 2)
        
        biases.append(bias)
        squared_errors.append(squared_error)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(biases, columns=["Bias_v", "Bias_a", "Bias_t"])
    df_se = pd.DataFrame(squared_errors, columns=["SE_v", "SE_a", "SE_t"])

    return df.mean(), df_se.mean()

# Run the simulation for N = 10, 40, 4000
if __name__ == "__main__":
    for N in [10, 40, 4000]:
        bias, squared_error = simulate_and_recover(N)
        print(f"\nResults for N={N}:")
        print("Average Bias:", bias)
        print("Average Squared Error:", squared_error)
