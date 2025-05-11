import matplotlib.pyplot as plt
from util.helper import *
from scipy.special import zeta

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
    alphas = get_step_sizes(n_iterations, X.T @ (M_2_sqrt @ M_2_sqrt) @ X, M_2_sqrt, step_type)
    
    # Initialize weights
    w_init = initialize_weights(X_hat, initialization)
    
    # Store results from multiple simulations
    all_diffs = np.zeros((n_simulations, n_iterations, n_features, 1))
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
            I = np.eye(n_features)
            # This is the update rule for the weights from equation (3) in the paper
            w = (I - alphas[k] * X.T @ D_k_squared @ X) @ w + alphas[k] * X.T @ D_k_squared @ Y
            all_diffs[sim, k] = w - w_hat
            all_second_diffs[sim, k] = (w - w_hat) @ (w - w_hat).T

    # Compute first moment (mean)
    mean_weights = np.mean(all_diffs, axis=0)
    first_moment = np.linalg.norm(mean_weights, axis=(1,2), ord=2)
    
    all_second_means = np.mean(all_second_diffs, axis=0)
    second_moment = np.linalg.norm(all_second_means, axis=(1,2), ord=2)
    second_moment_diff = np.zeros(n_iterations)

    for k in range(n_iterations):
        second_moment_diff[k] = np.linalg.norm(all_second_means[k] - compute_S_alpha(all_second_means[k], X, Y, w_hat, alphas[k], M_2_sqrt), ord=2)

    # Compute theoretical bounds
    X_norm = np.linalg.norm(X, ord=2)
    norm_sigma_d = np.linalg.norm((M_2_sqrt @ M_2_sqrt) @ (M_2_sqrt @ M_2_sqrt).T, ord=2)
    singular_value = get_minimum_nonzero_singular_value(X.T @ (M_2_sqrt @ M_2_sqrt) @ X)
    alpha = 0.9 * singular_value / (singular_value**2 + X_norm**4 * norm_sigma_d)
    euler_gamma = 0.5772156649
    sigma_min = get_minimum_nonzero_singular_value(X.T @ (M_2_sqrt @ M_2_sqrt) @ X)
    first_moment_bound = np.zeros(n_iterations)
    second_moment_bound = np.zeros(n_iterations)
    second_moment_diff_bound = np.zeros(n_iterations)
    
    C_0 = np.linalg.norm((w_init - w_hat) @ (w_init - w_hat).T, ord=2) + 2 * X_norm**3 * norm_sigma_d * np.linalg.norm(Y - X @ w_hat, ord=2) * np.linalg.norm(w_init - w_hat, ord=2)
    C_1 = C_0 * (1+alpha * (np.pi)**2/6) + X_norm**2 * norm_sigma_d * np.linalg.norm(Y - X @ w_hat, ord=2)**2 * (np.e)**(alpha * sigma_min * euler_gamma) * alpha * zeta(2-alpha * sigma_min)
    # Compute the product term for each k
    for k in range(0, n_iterations):
        product = 1.0
        for l in range(k):
            product *= (1 - alphas[l] * sigma_min)
            #print ("This is the product: ", product, "at iteration: ", k)
        first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
        print("In the first moment, the product is: ", product, "and the norm is: ", np.linalg.norm(w_init - w_hat, ord=2), "at iteration: ", k)
        # second_moment_bound[k] = 2 * alphas[k]**2 * np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(Y - X @ w_hat, ord=2)
        second_moment_diff_bound[k] = 2 * alphas[k]**2 * X_norm**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(Y - X @ w_hat, ord=2)
        #print("In the second moment, the product is: ", 2 * alphas[k]**2 * np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product, "and the weight norm is: ", np.linalg.norm(w_init - w_hat, ord=2), "and the residual norm is: ", np.linalg.norm(Y - X @ w_hat, ord=2), "at iteration: ", k)
        second_moment_bound[k] = C_1 * (1/k**(alpha * sigma_min))
    
    return first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10
    n_features = 200
    
    # Generate random data matrix X
    X = np.random.randn(n_samples, n_features)
    print("This is I - XX^+: ", np.linalg.norm(np.eye(n_features) - np.linalg.pinv(X) @ X, ord=2))

    G = np.random.randn(n_samples, n_samples)
    # do a QR decomposition
    Q, R = np.linalg.qr(G)

    A = np.random.randn(n_features, n_features)
    S,T = np.linalg.qr(A)

    # Generate Cauchy distributed values and scale them down
    sigma = np.random.randn(n_samples, n_features)
    # Set diagonal elements to zero with probability p
    p = 0.5
    diagonal_mask = np.random.binomial(1, p, size=n_samples)
    for i in range(n_samples):
        sigma[i,i] *= diagonal_mask[i]

    X = Q @ sigma @ S
    print("This is X - XX^+: ", np.linalg.norm(X - X @ np.linalg.pinv(X) @ X, ord=2))
    
    # Generate true weights (sparse) I am assuming that the true weights are sparse, since we are working with an over-parameterized model
    w_true = np.random.randn(n_features, 1)
    noise = 100 * np.random.randn(n_samples, 1)
    Y = X @ w_true + noise
    
    # Run simulations with different step types
    step_types = ['diminishing']
    
    for i, step_type in enumerate(step_types):
        first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound = simulate_weighted_gd(
            X, Y, n_iterations=100, step_type=step_type,
            initialization='orthogonal', weight_distribution='bernoulli', p=0.2, n_simulations=100)
        
        # Plot the convergence of the iterates to the expected solution
        # --- First plot: First moment convergence ---
        plt.figure(figsize=(8, 5))
        plt.plot(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[\hat w_{k+1} - \hat{w}]\|_2$')
        plt.plot(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        plt.title(f'First Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('First moment')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Second plot: Second moment vs operator S(.) ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment_diff, 'b-', linewidth=2,
                label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        plt.plot(second_moment_diff_bound, 'r--', linewidth=2, label='Second moment bound (Lemma 3.3)')
        plt.title(f'Second Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment difference')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Third plot: Norm of second moment ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment, 'b-', linewidth=2,
                label = r'$\|\mathbb{E}_{D}[(\hat{w}_{k+1} - \hat{w})(\hat{w}_{k+1} - \hat{w})^\top]\|_2$')
        plt.plot(second_moment_bound, 'r--', linewidth=2, label='Second moment bound (Lemma 3.3)')
        plt.title(f'Norm of the Second Moment ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 