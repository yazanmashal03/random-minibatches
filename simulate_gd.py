import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, svdvals

def get_M2_sqrt(weight_distribution, p, n_samples, sensitivities=None):
    """
    Get the square root of M2 matrix (E[D^2]) based on the weight distribution.
    For importance sampling, M2 is diagonal with entries being the sensitivities.
    """
    if weight_distribution == 'uniform':
        val = np.sqrt(1/3)
        return val * np.eye(n_samples)
    elif weight_distribution == 'bernoulli':
        val = np.sqrt(p)
        return val * np.eye(n_samples)
    elif weight_distribution == 'binary':
        val = np.sqrt(0.5)
        return val * np.eye(n_samples)
    elif weight_distribution == 'importance':
        if sensitivities is None:
            raise ValueError("Sensitivities must be provided for importance sampling")
        # For importance sampling, M2 = diag(sensitivities) since E[D^2] = P(D=1) = sensitivities
        return np.sqrt(np.diag(sensitivities))
    else:
        raise ValueError(f"Unknown weight distribution: {weight_distribution}")

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
    This uses the Gram-Schmidt process to orthogonalize the initial guess
    """
    n_samples, n_features = X.shape
    
    if initialization == 'orthogonal':
        ker_X = null_space(X)
        
        if ker_X.size > 0:
            w = np.random.randn(n_features)
            for v in ker_X.T:
                # Project w onto the orthogonal complement of ker(X).
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
    n_samples, n_features = X.shape
    I = np.eye(n_features)

    # Convert vector A to diagonal matrix if it's 1D
    if A.ndim == 1:
        A = np.diag(A)
    
    # First term: (I - alpha X_hat)A(I - alpha X_hat)
    term1 = (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X) @ A @ (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X)
    
    # Second term components
    residual = Y - X @ w_hat
    XAXt = X @ A.reshape(n_features, n_features) @ X.T
    residual_outer = np.outer(residual, residual)
    
    # Element-wise multiplication with Sigma_D (represented by M2_sqrt^2)
    Sigma_D = M2_sqrt @ M2_sqrt
    hadamard_term = Sigma_D * (XAXt + residual_outer)
    
    # Complete second term
    term2 = (alpha**2) * X.T @ hadamard_term @ X
    
    return term1 + term2

def get_D_matrix(n_samples, weight_distribution, p=None, sensitivities=None):
    """
    Generate diagonal matrix D based on the specified distribution
    """
    if weight_distribution == 'uniform':
        D_ii = np.random.uniform(0, 1, size=n_samples)
    elif weight_distribution == 'bernoulli':
        D_ii = np.random.binomial(1, p, size=n_samples)
    elif weight_distribution == 'binary':
        indices = np.random.choice(n_samples, size=n_samples//2, replace=False)
        D_ii = np.zeros(n_samples)
        D_ii[indices] = 1
    elif weight_distribution == 'importance':
        if sensitivities is None:
            raise ValueError("Sensitivities must be provided for importance sampling")
        D_ii = np.random.binomial(1, p=sensitivities, size=n_samples)
    else:
        raise ValueError(f"Unknown weight distribution: {weight_distribution}")
    
    return np.diag(D_ii)

def compute_sensitivities(X, Y, w):
    """
    Compute sensitivities for importance sampling based on current residuals
    """
    residuals = Y - X @ w
    sensitivities = residuals**2
    return sensitivities / np.sum(sensitivities)  # normalize to get probabilities

def simulate_weighted_gd(X, Y, n_iterations=100, step_type='constant', initialization='orthogonal', 
                        weight_distribution='bernoulli', p=0.2, n_simulations=50):
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
    all_weights = np.zeros((n_simulations, n_iterations + 1, n_features))
    
    # Run multiple simulations
    for sim in range(n_simulations):
        w = w_init.copy()
        all_weights[sim, 0] = w
        
        for k in range(n_iterations):
            # Compute sensitivities if using importance sampling
            sensitivities = None
            if weight_distribution == 'importance':
                sensitivities = compute_sensitivities(X, Y, w)
            
            # Create diagonal matrix based on distribution
            D_k = get_D_matrix(n_samples, weight_distribution, p, sensitivities)
            
            # Update weights
            D_k_squared = D_k @ D_k
            I_features = np.eye(n_features)
            w = (I_features - alphas[k] * X.T @ D_k_squared @ X) @ w + alphas[k] * X.T @ D_k_squared @ Y
            all_weights[sim, k+1] = w

    # Compute differences from the true solution
    all_diffs = all_weights - w_hat

    # Compute first moment (mean)
    mean_weights = np.mean(all_diffs, axis=0)
    first_moment = np.linalg.norm(mean_weights, axis=1, ord=2)[:n_iterations]  # Truncate to match bounds
    
    # Compute empirical second moment
    all_second_diffs = np.zeros((n_simulations, n_iterations, n_features))

    for sim in range(n_simulations):
        for k in range(n_iterations):
            diff = all_diffs[sim, k]
            all_second_diffs[sim, k] = diff * diff
    
    all_second_means = np.mean(all_second_diffs, axis=0)
    all_S_alpha = np.zeros((n_iterations, n_features, n_features))
    
    second_moment = np.zeros(n_iterations)
    for k in range(n_iterations):
        all_S_alpha[k] = compute_S_alpha(all_second_means[k], X, Y, w_hat, alphas[k], M_2_sqrt)
        second_moment[k] = np.linalg.norm(np.diag(all_second_means[k]) - all_S_alpha[k], ord=2)

    # Compute theoretical bounds
    sigma_min = get_minimum_nonzero_singular_value(X.T @ (M_2_sqrt @ M_2_sqrt) @ X)
    norm_sigma_d = np.linalg.norm(M_2_sqrt @ M_2_sqrt, ord=2)
    first_moment_bound = np.zeros(n_iterations)
    second_moment_bound = np.zeros(n_iterations)
    first_moment_bound[0] = np.linalg.norm(w_init - w_hat, ord=2)  # Initial difference
    second_moment_bound[0] = np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(Y - X @ w_hat, ord=2)  # Initial bound
    
    # Compute the product term for each k
    for k in range(1, n_iterations):
        product = 1.0
        for l in range(k):
            product *= (1 - alphas[l] * sigma_min)
        first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
        print("This is first moment bound: ", first_moment_bound[k], "at iteration: ", k)
        second_moment_bound[k] = 2 * alphas[k]**2 * np.linalg.norm(X, ord=2)**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(Y - X @ w_hat, ord=2)
        print("This is second moment bound: ", second_moment_bound[k], "at iteration: ", k)
    
    return first_moment, first_moment_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 40
    
    # Generate random data matrix X
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights (sparse) I am assuming that the true weights are sparse, since we are working with an over-parameterized model
    w_true = np.zeros(n_features)
    w_true[:10] = np.random.randn(10)
    
    # Generate target values
    noise = 0.1 * np.random.randn(n_samples)
    Y = X @ w_true + noise
    
    # Run simulations with different step types
    step_types = ['diminishing']
    
    plt.figure(figsize=(15, 12))
    
    for i, step_type in enumerate(step_types):
        first_moment, first_moment_bound, second_moment, second_moment_bound = simulate_weighted_gd(
            X, Y, n_iterations=1500, step_type=step_type,
            initialization='orthogonal', weight_distribution='bernoulli', p=0.2)
        
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
    plt.savefig('Figures/lemma_3_2_and_3_3.png')
    plt.close()

if __name__ == "__main__":
    main() 