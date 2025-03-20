#Used ChatGPT throughout to debug
import sys
from src.simulate import simulate_data
from src.recover import inverse_equations

def simulate_and_recover(N, iterations=1000):
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        try:
            a_true, v_true, t_true, R_obs, M_obs, V_obs = simulate_data(N)
            v_est, a_est, t_est = inverse_equations(R_obs, M_obs, V_obs)

            biases.append((v_true - v_est, a_true - a_est, t_true - t_est))
            squared_errors.append((biases[-1][0] ** 2, biases[-1][1] ** 2, biases[-1][2] ** 2))
        except Exception as e:
            print(f"Simulation failed: {e}")
            sys.exit(1)

    mean_bias = tuple(sum(col) / len(col) for col in zip(*biases))
    mean_se = tuple(sum(col) / len(col) for col in zip(*squared_errors))

    return mean_bias, mean_se

if __name__ == "__main__":
    for N in [10, 40, 4000]:
        bias, squared_error = simulate_and_recover(N)
        print(f"\nResults for N={N}:")
        print("Average Bias:", bias)
        print("Average Squared Error:", squared_error)
