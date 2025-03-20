#Used ChatGPT and ZotGPT throughout to debug/troubleshoot
import numpy as np

A_RANGE = (0.5, 2.0)
V_RANGE = (0.5, 2.0)
T_RANGE = (0.1, 0.5)

def generate_parameters():
    """Generates random parameters within realistic ranges."""
    a = np.random.uniform(*A_RANGE)
    v = np.random.uniform(*V_RANGE)
    t = np.random.uniform(*T_RANGE)
    return a, v, t

def simulate_data(a, v, t, N):
    """Simulates response times and accuracy based on EZ diffusion model."""
    y = np.exp(-v * a)
    R = 1 / (1 + y)
    M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V = (a / (2 * v**2)) * ((1 - y**2) / (1 + y)**2)
    
    # Generate N samples of RT and accuracy
    accuracy = np.random.binomial(n=1, p=R, size=N)
    rt = np.random.normal(loc=M, scale=np.sqrt(V), size=N)
    rt = np.maximum(rt, t)  # Ensure RTs are not less than non-decision time
    return accuracy, rt

def recover_parameters(accuracy, rt):
    """Recovers EZ diffusion model parameters from simulated data."""
    R_obs = np.mean(accuracy)
    M_obs = np.mean(rt)
    V_obs = np.var(rt)

    # Prevent invalid log operations
    R_obs = np.clip(R_obs, 0.001, 0.999)
    L = np.log(R_obs / (1 - R_obs))

    # Ensure variance is large enough to avoid instability
    if V_obs < 1e-6 or np.isnan(L) or np.isinf(L):
        return np.nan, np.nan, np.nan

    try:
        s = 1.0  # scaling parameter
        v_est = np.sign(R_obs - 0.5) * s * L / (M_obs * L - s**2 * V_obs / 2)
        a_est = s**2 * L / v_est
        y = -v_est * a_est
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(y)) / (1 + np.exp(y)))

        # Ensure all parameters are positive and within reasonable bounds
        if a_est <= 0 or a_est > 5 or abs(v_est) > 5 or t_est < 0 or t_est > M_obs:
            return np.nan, np.nan, np.nan

    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan, np.nan, np.nan

    return a_est, v_est, t_est

def run_simulation(N, iterations=5000):
    """Runs the simulate-and-recover process for given N and iterations."""
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        a_true, v_true, t_true = generate_parameters()
        acc, rt = simulate_data(a_true, v_true, t_true, N)
        a_est, v_est, t_est = recover_parameters(acc, rt)

        # Ignore NaN values or unrealistic recoveries
        if (np.isnan(a_est) or np.isnan(v_est) or np.isnan(t_est) or
            a_est <= 0 or a_est > 5 or abs(v_est) > 5 or t_est < 0 or t_est > np.mean(rt)):
            continue
        
        bias = (a_est - a_true, v_est - v_true, t_est - t_true)
        se = (bias[0]**2, bias[1]**2, bias[2]**2)
        
        # Filter out extreme recoveries
        if max(abs(b) for b in bias) > 2:
            continue
        
        biases.append(bias)
        squared_errors.append(se)
    
    mean_bias = np.mean(biases, axis=0) if biases else (np.nan, np.nan, np.nan)
    mean_se = np.mean(squared_errors, axis=0) if squared_errors else (np.nan, np.nan, np.nan)
    
    return N, mean_bias, mean_se

if __name__ == "__main__":
    results = []
    for N in [10, 40, 4000]:
        result = run_simulation(N)
        results.append(result)
    
    with open("results.txt", "w") as f:
        for res in results:
            f.write(f"N: {res[0]}, Bias: {res[1]}, Squared Error: {res[2]}\n")

    print("Simulation complete. Results saved in results.txt.")

