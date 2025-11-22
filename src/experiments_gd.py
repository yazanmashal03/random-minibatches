import os
import numpy as np
from utils.helper import plot_simulation_results
from weighted_gd import WeightedGD

def run_experiment(n_samples, n_features, weight_distribution, results_dir):
    np.random.seed(59)
    n_iterations = 500
    n_simulations = 10
    initialization = 'orthogonal'
    alpha = 0.65
    step_type = 'constant'
    batch_size = "random"

    # Data generation (same as in your main)
    G = np.random.randn(n_samples, n_samples)
    Q, R = np.linalg.qr(G)
    A = np.random.randn(n_features, n_features)
    S, T = np.linalg.qr(A)
    sigma = np.zeros((n_samples, n_features))
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p=0.6) * np.abs(1 + np.random.randn())
    X = Q @ sigma @ S
    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    # Generate weights based on weight distribution
    if weight_distribution == 'importance':
        weights = np.linalg.norm(X, axis=1)
    elif weight_distribution == 'bernoulli':
        weights = np.ones(n_samples)  # Equal weights for uniform sampling
    else:
        weights = np.ones(n_samples)  # Default to uniform


    wg = WeightedGD(X, Y, alpha, n_iterations, step_type, initialization, weight_distribution, weights, batch_size, n_simulations)
    
    try:
        first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist = wg.simulate_weighted_gd()
        # Save plot
        if first_moment is not None and first_moment_bound is not None and second_moment is not None and second_moment_bound is not None:
            plot_path = os.path.join(results_dir, f"sim_n{n_samples}_d{n_features}_wdist_{weight_distribution}.png")
            try:
                plot_simulation_results(first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist, step_type, save_path=plot_path)
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
    except Exception as e:
        print(f"Exception during simulation: {e}")
        # Save a plot that says "Error converging"
        plot_path = os.path.join(results_dir, f"sim_n{n_samples}_d{n_features}_wdist_{weight_distribution}.png")
        try:
            plot_simulation_results(None, None, None, None, None, step_type, save_path=plot_path)
            print(f"Error plot saved successfully to: {plot_path}")
        except Exception as plot_error:
            print(f"Error saving error plot: {plot_error}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "figures")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")
    print(f"Current working directory: {os.getcwd()}")

    n_samples_list = [10, 50, 100]
    n_features_list = [10, 50, 100, 200]
    weight_distributions = ['bernoulli', 'noiseless']

    for n_samples in n_samples_list:
        for n_features in n_features_list:
            for wdist in weight_distributions:
                print(f"Running: n_samples={n_samples}, n_features={n_features}, wdist={wdist}")
                run_experiment(n_samples, n_features, wdist, results_dir)
