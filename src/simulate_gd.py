import matplotlib.pyplot as plt
from utils.helper import *
from weighted_gd import WeightedGD
from scipy.stats import cauchy

def main():
    # Generate synthetic data
    np.random.seed(34)
    n_samples = 10
    n_features = 100

    n_iterations = 500
    n_simulations = 10
    p_succ = 0.2
    weight_distribution = 'uniform'
    initialization = 'orthogonal'
    alpha = 0.65 # this is the step size for the constant step size
    step_type = 'constant'

    # Generate random data matrix X using the QR decomposition
    G = np.random.randn(n_samples, n_samples)

    # do a QR decomposition
    Q, R = np.linalg.qr(G)

    A = np.random.randn(n_features, n_features)
    S,T = np.linalg.qr(A)

    # Generate the eigenvalue matrix sigma
    sigma = np.zeros((n_samples, n_features))

    # Set diagonal elements to random normal with probability p
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p=0.6) * 1 * np.random.randn()

    X = Q @ sigma @ S
    print("This is I - XX^+: ", np.linalg.norm(np.eye(n_samples) - X @ np.linalg.pinv(X), ord=2))
    
    # Generate true weights
    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    wg = WeightedGD(X, Y, w_true, alpha, n_iterations, step_type, initialization, weight_distribution, p_succ, n_simulations)
    
    first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound = wg.simulate_weighted_gd()
    plot_simulation_results(first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound, step_type)

if __name__ == "__main__":
    main() 