import matplotlib.pyplot as plt
from utils.helper import *
from weighted_gd import WeightedGD

def main():
    # Generate synthetic data
    np.random.seed(31012)
    n_samples = 10
    n_features = 50

    n_iterations = 200
    n_simulations = 1000
    batch_size = 1 # set this to some integer to specify batch-size, or choose 'random' to get independently active data points
    weight_distribution = 'bernoulli'
    initialization = 'orthogonal'
    alpha = 0.01
    step_type = 'constant'

    # Generate random data matrix X using the QR decomposition
    G = np.random.randn(n_samples, n_samples)

    # do a QR decomposition
    Q, R = np.linalg.qr(G)

    A = np.random.randn(n_features, n_features)
    S, T = np.linalg.qr(A)

    # Generate the eigenvalue matrix sigma
    sigma = np.zeros((n_samples, n_features))

    # Set diagonal elements to random normal with probability p
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p = 0.3) * np.abs(5 + np.random.randn())

    X = Q @ sigma @ S
    X = X # / np.linalg.norm(X, ord=2)

    # Generate true weights
    w_true = 0.5 * np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    # weights for importance sampling (equal entries leads to uniform sampling)
    # weights = np.ones(n_samples)
    weights = np.linalg.norm(X, axis = 1)

    wg = WeightedGD(X, Y, alpha, n_iterations, step_type, initialization, weight_distribution, weights, batch_size, n_simulations)

    first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist = wg.simulate_weighted_gd()
    plot_simulation_results(first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist, step_type)

if __name__ == "__main__":
    main()
