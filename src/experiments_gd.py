import os
import numpy as np
from utils.helper import plot_simulation_results
from weighted_gd import WeightedGD

def run_experiment(n_samples, n_features, weight_distribution, results_dir):
    np.random.seed(34)
    n_iterations = 500
    n_simulations = 10
    p_succ = 0.7
    initialization = 'orthogonal'
    alpha = 0.65
    step_type = 'constant'

    # Data generation (same as in your main)
    G = np.random.randn(n_samples, n_samples)
    Q, R = np.linalg.qr(G)
    A = np.random.randn(n_features, n_features)
    S, T = np.linalg.qr(A)
    sigma = np.zeros((n_samples, n_features))
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p=0.6) * 1 * np.random.randn()
    X = Q @ sigma @ S
    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    wg = WeightedGD(X, Y, w_true, alpha, n_iterations, step_type, initialization, weight_distribution, p_succ, n_simulations)
    first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound = wg.simulate_weighted_gd()

    # Save plot
    if first_moment is not None and first_moment_bound is not None and second_moment_diff is not None and second_moment_diff_bound is not None and second_moment is not None and second_moment_bound is not None:
        plot_path = os.path.join(results_dir, f"sim_n{n_samples}_d{n_features}_wdist_{weight_distribution}.png")
        try:
            plot_simulation_results(first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound, step_type, save_path=plot_path)
            print(f"Plot saved successfully to: {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Results directory: {results_dir}")
            print(f"Plot path: {plot_path}")
    else:
        # Save a plot that says "Error converging"
        plot_path = os.path.join(results_dir, f"sim_n{n_samples}_d{n_features}_wdist_{weight_distribution}.png")
        plot_simulation_results(None, None, None, None, None, None, step_type, save_path=plot_path)
        print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    results_dir = "../figures"
    os.makedirs(results_dir, exist_ok=True)

    n_samples_list = [10, 50, 100]
    n_features_list = [10, 50, 100, 200]
    weight_distributions = ['uniform', 'bernoulli', 'importance']

    for n_samples in n_samples_list:
        for n_features in n_features_list:
            for wdist in weight_distributions:
                print(f"Running: n_samples={n_samples}, n_features={n_features}, wdist={wdist}")
                run_experiment(n_samples, n_features, wdist, results_dir)
