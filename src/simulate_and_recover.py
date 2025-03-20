import numpy as np

# Parameter ranges
A_RANGE = (0.5, 2.0)  # Boundary separation
V_RANGE = (0.5, 2.0)  # Drift rate
T_RANGE = (0.1, 0.5)  # Non-decision time

# Function to generate parameters
def generate_parameters():
    """Generates random parameters within realistic ranges."""
    a = np.random.uniform(*A_RANGE)
    v = np.random.uniform(*V_RANGE)
    t = np.random.uniform(*T_RANGE)
    return a, v, t

# Forward EZ equations
def simulate_data(a, v, t, N):
    """Simulates response times and accuracy based on EZ diffusion model."""
    y = np.exp(-a * v)
    R = 1 / (y + 1)
    M = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
    
    # Generate N samples of RT and accuracy
    accuracy = np.random.binomial(n=1, p=R, size=N)
    rt = np.random.normal(loc=M, scale=max(np.sqrt(V), 1e-6), size=N)  # Prevent zero std deviation
    return accuracy, rt

# Inverse EZ equations to recover parameters
def recover_parameters(accuracy, rt):
    """Recovers EZ diffusion model parameters from simulated data."""
    R_obs = np.mean(accuracy)
    M_obs = np.mean(rt)
    V_obs = np.var(rt)

    # Prevent invalid log operations (log(0) or log(negative values))
    R_obs = np.clip(R_obs, 0.05, 0.95)  
    L = np.log(R_obs / (1 - R_obs))

    # Ensure variance is large enough to avoid instability
    if V_obs < 1e-4 or np.isnan(L) or np.isinf(L):
        return np.nan, np.nan, np.nan  # Return NaN if calculations are unstable

    try:
        # Adjust v_est to be more stable
        v_est = np.sign(R_obs - 0.5) * np.sqrt(abs(L)) / (np.sqrt(abs(V_obs)) + 1e-6)
        
        # Adjust a_est calculation
        a_est = abs(L) / (abs(v_est) + 1e-6)  # Ensure division by zero does not happen
        
        # Ensure numerical stability in t_est
        t_est = M_obs - (a_est / (2 * v_est + 1e-6)) * ((1 - np.exp(-abs(v_est * a_est))) / (1 + np.exp(-abs(v_est * a_est))))

    except (FloatingPointError, ValueError):
        return np.nan, np.nan, np.nan  # Handle numerical errors gracefully

    return a_est, v_est, t_est

# Main function for simulate-and-recover experiment
def run_simulation(N, iterations=1000):
    """Runs the simulate-and-recover process for given N and iterations."""
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        a_true, v_true, t_true = generate_parameters()
        acc, rt = simulate_data(a_true, v_true, t_true, N)
        a_est, v_est, t_est = recover_parameters(acc, rt)

        # Ignore NaN values (caused by numerical instability)
        if np.isnan(a_est) or np.isnan(v_est) or np.isnan(t_est):
            continue
        
        bias = (a_est - a_true, v_est - v_true, t_est - t_true)
        se = (bias[0]**2, bias[1]**2, bias[2]**2)
        
        biases.append(bias)
        squared_errors.append(se)
    
    # Compute mean bias and squared error
    mean_bias = np.mean(biases, axis=0) if biases else (np.nan, np.nan, np.nan)
    mean_se = np.mean(squared_errors, axis=0) if squared_errors else (np.nan, np.nan, np.nan)
    
    return N, mean_bias, mean_se

if __name__ == "__main__":
    results = []
    for N in [10, 40, 4000]:
        result = run_simulation(N)
        results.append(result)
    
    # Save results to a text file
    with open("results.txt", "w") as f:
        for res in results:
            f.write(f"N: {res[0]}, Bias: {res[1]}, Squared Error: {res[2]}\n")

    print("Simulation complete. Results saved in results.txt.")
