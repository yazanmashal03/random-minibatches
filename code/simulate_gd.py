import matplotlib.pyplot as plt
from util.helper import *

def simulate_weighted_gd(X, Y, n_iterations=1000, step_type='constant', initialization='orthogonal', 
                        weight_distribution='bernoulli', p=0.2, n_simulations=100):
    """
    Simulate randomly weighted gradient descent and compute moments
    """
    n_samples, n_features = X.shape
    
    # Compute initial sensitivities if using importance sampling
    sensitivities = None
    if weight_distribution == 'importance':
        # Initialize w randomly to compute initial sensitivities
        w_init_sens = np.random.randn(n_features)
        sensitivities = compute_sensitivities(X, Y, w_init_sens)
    
    # Used to compute the true solution and satisfy the assumptions of the theorems.
    M_2_sqrt = get_M2_sqrt(weight_distribution, p, n_samples, sensitivities)
    X_hat = M_2_sqrt @ X
    Y_hat = M_2_sqrt @ Y
    w_hat = np.linalg.pinv(X_hat) @ Y_hat

    alphas = get_step_sizes(n_iterations, X.T @ (M_2_sqrt @ M_2_sqrt) @ X, step_type)
    
    # Initialize weights
    w_init = initialize_weights(X_hat, initialization)
    
    # Store results from multiple simulations
    all_diffs = np.zeros((n_simulations, n_iterations, n_features))
    all_second_diffs = np.zeros((n_simulations, n_iterations, n_features, n_features))
    
    # Run multiple simulations
    for sim in range(n_simulations):
        w = w_init.copy()
        
        for k in range(n_iterations):
            sensitivities = None
            if weight_distribution == 'importance':
                sensitivities = compute_sensitivities(X, Y, w)
            
            D_k = get_D_matrix(n_samples, weight_distribution, p, sensitivities)
            
            # Update weights
            D_k_squared = D_k @ D_k
            I_features = np.eye(n_features)
            w = (I_features - alphas[k] * X.T @ D_k_squared @ X) @ w + alphas[k] * X.T @ D_k_squared @ Y
            all_diffs[sim, k] = w - w_hat
            all_second_diffs[sim, k] = np.outer((w - w_hat), (w - w_hat))

    # Compute first moment (mean)
    mean_weights = np.mean(all_diffs, axis=0)
    first_moment = np.linalg.norm(mean_weights, axis=1, ord=2)
    
    all_second_means = np.mean(all_second_diffs, axis=0)
    
    second_moment = np.zeros(n_iterations)
    for k in range(n_iterations):
        second_moment[k] = np.linalg.norm(all_second_means[k] - compute_S_alpha(all_second_means[k], X, Y, w_hat, alphas[k], M_2_sqrt), ord=2)

    # Compute theoretical bounds
    sigma_min = get_minimum_nonzero_singular_value(X.T @ (M_2_sqrt @ M_2_sqrt) @ X)
    norm_sigma_d = np.linalg.norm(M_2_sqrt @ M_2_sqrt, ord=2)
    first_moment_bound = np.zeros(n_iterations)
    second_moment_bound = np.zeros(n_iterations)
    
    # Compute the product term for each k
    for k in range(0, n_iterations):
        product = 1.0
        for l in range(k):
            product *= (1 - alphas[l] * sigma_min)
            #print ("This is the product: ", product, "at iteration: ", k)
        first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
        print("In the first moment, the product is: ", product, "and the norm is: ", np.linalg.norm(w_init - w_hat, ord=2), "at iteration: ", k)
        # second_moment_bound[k] = 2 * alphas[k]**2 * np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(Y - X @ w_hat, ord=2)
        second_moment_bound[k] = np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2)
        print("In the second moment, the product is: ", 2 * alphas[k]**2 * np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product, "and the weight norm is: ", np.linalg.norm(w_init - w_hat, ord=2), "and the residual norm is: ", np.linalg.norm(Y - X @ w_hat, ord=2), "at iteration: ", k)
    
    return first_moment, first_moment_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10
    n_features = 200
    
    # Generate random data matrix X
    X = np.random.randn(n_samples, n_features)
    print("This is I - XX^+: ", np.linalg.norm(np.eye(n_features) - np.linalg.pinv(X) @ X, ord=2))

    # G = np.random.randn(n_samples, n_samples)
    # # do a QR decomposition
    # Q, R = np.linalg.qr(G)

    # A = np.random.randn(n_features, n_features)
    # S,T = np.linalg.qr(A)

    # # Generate Cauchy distributed values and scale them down
    # sigma = np.random.randn(n_samples, n_features)
    # # Set diagonal elements to zero with probability p
    # p = 0.5
    # diagonal_mask = np.random.binomial(1, p, size=n_samples)
    # for i in range(n_samples):
    #     sigma[i,i] *= diagonal_mask[i]

    # X = Q @ sigma @ S
    # print("This is X - XX^+: ", np.linalg.norm(X - X @ np.linalg.pinv(X) @ X, ord=2))
    
    # Generate true weights (sparse) I am assuming that the true weights are sparse, since we are working with an over-parameterized model
    w_true = np.random.randn(n_features)
    noise = 100 * np.random.randn(n_samples)
    Y = X @ w_true + noise
    
    # Run simulations with different step types
    step_types = ['diminishing']
    
    plt.figure(figsize=(15, 12))
    
    for i, step_type in enumerate(step_types):
        first_moment, first_moment_bound, second_moment, second_moment_bound = simulate_weighted_gd(
            X, Y, n_iterations=5000, step_type=step_type,
            initialization='orthogonal', weight_distribution='importance', p=0.2, n_simulations=1)
        
        # Plot first moment results
        plt.subplot(2, 1, i+1)
        plt.plot(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[\hat w_{k+1} - \hat{w}]\|_2$')
        plt.plot(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        plt.title(f'First Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('First moment')
        plt.legend()
        plt.grid(True)
        
        # Plot second moment results
        plt.subplot(2, 1, i+2)
        plt.plot(second_moment, 'b-', linewidth=2,
                label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        plt.plot(second_moment_bound, 'r--', linewidth=2,
                label='Second moment bound (Lemma 3.3)')
        plt.title(f'Second Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Spectral norm of second moment')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Figures/simulation_results.png')
    plt.close()

if __name__ == "__main__":
    main() 