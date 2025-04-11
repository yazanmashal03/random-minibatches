import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, svdvals

def get_M2_sqrt(weight_distribution, p, n_samples):
    if weight_distribution == 'uniform':
        val = np.sqrt(1/3)
    elif weight_distribution == 'bernoulli':
        val = np.sqrt(p)
    else:  # binary
        val = np.sqrt(0.5)
    return val * np.eye(n_samples)

def get_step_sizes(n_iterations, X, step_type='constant'):
    """
    Generate step sizes satisfying Assumption 3.1(a)
    
    Parameters:
    n_iterations: number of iterations
    X: data matrix
    step_type: 'constant' or 'diminishing'
    """
    X_norm = np.linalg.norm(X, ord=2)  # spectral norm
    
    if step_type == 'constant':
        alpha = 0.01
        return alpha * np.ones(n_iterations)
    else:
        c = 0.9 / X_norm
        return c / np.arange(1, n_iterations + 1)

def initialize_weights(X, initialization='orthogonal'):
    """
    Initialize weights satisfying Assumption 3.1(b):
    Initial guess should lie in the orthogonal complement of ker(X)
    """
    n_samples, n_features = X.shape
    
    if initialization == 'orthogonal':
        ker_X = null_space(X)
        
        if ker_X.size > 0:
            w = np.random.randn(n_features)
            for v in ker_X.T:
                w = w - (w @ v) * v
            w = w / np.linalg.norm(w)
        else:
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
    else:
        if initialization == 'zero':
            w = np.zeros(n_features)
        else:
            w = np.random.randn(n_features) * 0.01
            
    return w

def get_minimum_nonzero_singular_value(X):
    """
    Compute the minimum non-zero singular value of X
    """
    singular_values = svdvals(X)
    nonzero_singular_values = singular_values[singular_values > 1e-10]
    return np.min(nonzero_singular_values) if len(nonzero_singular_values) > 0 else 0

def compute_S_alpha(A, X, Y, w_hat, alpha, M2_sqrt):
    """
    Compute the S_alpha operator from Lemma 3.3:
    S_alpha(A) = (I - alpha X_hat)A(I - alpha X_hat) + alpha^2 X^T(Sigma_D ○ (XAX^T + (Y-Xw_hat)(Y-Xw_hat)^T))X
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    X_hat = M2_sqrt @ X
    
    # First term: (I - alpha X_hat)A(I - alpha X_hat)
    term1 = (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X) @ A @ (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X)
    
    # Second term components
    residual = Y - X @ w_hat
    XAXt = X @ A @ X.T
    residual_outer = np.outer(residual, residual)
    
    # Element-wise multiplication with Sigma_D (represented by M2_sqrt^2)
    Sigma_D = M2_sqrt @ M2_sqrt
    hadamard_term = Sigma_D * (XAXt + residual_outer)
    
    # Complete second term
    term2 = (alpha**2) * X.T @ hadamard_term @ X
    
    return term1 + term2

def compute_second_moment_bound(X, Y, w_init, w_hat, alphas, weight_distribution, p, n_samples):
    """
    Compute the upper bound on the second moment from Lemma 3.3
    """
    # Compute required norms and values
    X_norm = np.linalg.norm(X, ord=2)
    M2_sqrt = get_M2_sqrt(weight_distribution, p, n_samples)
    Sigma_D_norm = np.linalg.norm(M2_sqrt @ M2_sqrt, ord=2)
    sigma_min_plus = get_minimum_nonzero_singular_value(X.T @ X)
    w1_diff_norm = np.linalg.norm(w_init - w_hat)
    residual_norm = np.linalg.norm(Y - X @ w_hat)
    
    n_iterations = len(alphas)
    rho_bound = np.zeros(n_iterations)
    
    # Compute cumulative sum of alphas for the exponential term
    cumsum_alphas = np.cumsum(alphas)
    
    # Compute bound for each iteration k ≥ 1
    for k in range(1, n_iterations):  # Start from k=1
        exp_term = np.exp(-sigma_min_plus * cumsum_alphas[k-1])
        rho_bound[k] = 2 * (alphas[k]**2) * (X_norm**3) * Sigma_D_norm * exp_term * w1_diff_norm * residual_norm
    
    return rho_bound

def simulate_weighted_gd(X, Y, n_iterations=100, step_type='constant', initialization='orthogonal', 
                        weight_distribution='uniform', p=0.2, n_simulations=50):
    """
    Simulate randomly weighted gradient descent and compute moments
    """
    n_samples, n_features = X.shape
    alphas = get_step_sizes(n_iterations, X, step_type)
    
    # Initialize weights
    w_init = initialize_weights(X, initialization)
    
    # Store results from multiple simulations
    all_weights = np.zeros((n_simulations, n_iterations + 1, n_features))
    
    # Run multiple simulations
    for sim in range(n_simulations):
        w = w_init.copy()
        all_weights[sim, 0] = w
        
        for k in range(n_iterations):
            # Create diagonal matrix
            if weight_distribution == 'uniform':
                D_k = np.diag(np.random.uniform(0, 1, size=n_samples))
            elif weight_distribution == 'bernoulli':
                D_k = np.diag(np.random.binomial(1, p, size=n_samples))
            else:  # binary
                indices = np.random.choice(n_samples, size=n_samples//2, replace=False)
                D_k = np.zeros((n_samples, n_samples))
                D_k[indices, indices] = 1
            
            # Update weights
            D_k_squared = D_k @ D_k
            I_features = np.eye(n_features)
            w = (I_features - alphas[k] * X.T @ D_k_squared @ X) @ w + alphas[k] * X.T @ D_k_squared @ Y
            all_weights[sim, k+1] = w
    
    # Compute the true solution
    M_2_sqrt = get_M2_sqrt(weight_distribution, p, n_samples)
    X_hat = M_2_sqrt @ X
    Y_hat = M_2_sqrt @ Y
    w_hat = np.linalg.pinv(X_hat) @ Y_hat

    # Compute differences from the true solution
    all_diffs = all_weights - w_hat

    # Compute first moment (mean)
    mean_weights = np.mean(all_diffs, axis=0)
    first_moment = np.linalg.norm(mean_weights, axis=1)
    
    # Compute empirical second moment
    second_moment = np.zeros(n_iterations + 1)
    second_moment_matrices = np.zeros((n_iterations + 1, n_features, n_features))
    
    for k in range(n_iterations + 1):
        diffs_k = all_diffs[:, k, :]  # (n_simulations, n_features)
        # Compute average outer product for current iteration
        second_moment_matrices[k] = np.mean([np.outer(diff, diff) for diff in diffs_k], axis=0)
        
        if k > 0:
            # Get the previous second moment matrix and apply S_alpha to it
            prev_moment = second_moment_matrices[k-1]
            S_applied = compute_S_alpha(prev_moment, X, Y, w_hat, alphas[k-1], M_2_sqrt)
            
            # Compute the difference and its Frobenius norm
            diff_matrix = second_moment_matrices[k] - S_applied
            second_moment[k] = np.linalg.norm(diff_matrix, 'fro')
    
    # First iteration has no previous moment to compare with
    second_moment[0] = np.linalg.norm(second_moment_matrices[0], 'fro')
    
    # Compute theoretical bounds
    first_moment_bound = np.exp(-get_minimum_nonzero_singular_value(X.T @ X) * np.cumsum(np.append(0, alphas))) * np.linalg.norm(w_init - w_hat)
    second_moment_bound = compute_second_moment_bound(X, Y, w_init, w_hat, alphas, weight_distribution, p, n_samples)
    
    return first_moment, first_moment_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 90
    
    # Generate random data matrix X
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights (sparse)
    w_true = np.zeros(n_features)
    w_true[:10] = np.random.randn(10)
    
    # Generate target values
    noise = 0.1 * np.random.randn(n_samples)
    Y = X @ w_true + noise
    
    # Run simulations with different step types
    step_types = ['constant', 'diminishing']
    
    plt.figure(figsize=(15, 12))
    
    for i, step_type in enumerate(step_types):
        # Run simulation
        first_moment, first_moment_bound, second_moment, second_moment_bound = simulate_weighted_gd(
            X, Y, n_iterations=1000, step_type=step_type,
            initialization='orthogonal', weight_distribution='uniform')
        
        # Plot first moment results
        plt.subplot(2, 2, i+1)
        plt.semilogy(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}[w_k - \hat{w}]\|$')
        plt.semilogy(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        plt.title(f'First Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('First moment')
        plt.legend()
        plt.grid(True)
        
        # Plot second moment results
        plt.subplot(2, 2, i+3)
        plt.semilogy(second_moment, 'b-', linewidth=2, 
                    label=r'$\|\mathbb{E}[(w_k - \hat{w})(w_k - \hat{w})^\top]\|_F$')
        plt.semilogy(second_moment_bound, 'r--', linewidth=2,
                    label='Second moment bound (Lemma 3.3)')
        plt.title(f'Second Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Frobenius norm of second moment')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lemma_3_2_and_3_3.png')
    plt.close()

if __name__ == "__main__":
    main() 