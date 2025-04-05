import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, svdvals

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
    Compute σ_min^+(X), the minimum non-zero singular value of X
    """
    singular_values = svdvals(X)
    nonzero_singular_values = singular_values[singular_values > 1e-10]
    return np.min(nonzero_singular_values) if len(nonzero_singular_values) > 0 else 0

def simulate_weighted_gd(X, Y, n_iterations=100, step_type='constant', initialization='orthogonal', 
                        weight_distribution='uniform', p=0.2, n_simulations=10):  # increased simulations for better mean estimate
    """
    Simulate randomly weighted gradient descent and compute consecutive weight differences
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
    
    # First compute mean weights across simulations
    mean_weights = np.mean(all_weights, axis=0)  # shape: (n_iterations + 1, n_features)
    
    # Then compute consecutive differences of means ||E[w_{k+1}] - E[w_k]||
    consecutive_diffs = np.linalg.norm(mean_weights[1:] - mean_weights[:-1], axis=1)
    
    # Compute upper bound from Lemma 3.2
    sigma_min_plus = get_minimum_nonzero_singular_value(X)
    cumsum_alphas = np.cumsum(alphas)
    upper_bound = np.exp(-sigma_min_plus * cumsum_alphas) * np.linalg.norm(w_init - mean_weights[-1])
    
    return consecutive_diffs, upper_bound

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 200
    
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
    
    plt.figure(figsize=(15, 6))
    
    for i, step_type in enumerate(step_types):
        # Run simulation
        consecutive_diffs, upper_bound = simulate_weighted_gd(
            X, Y, n_iterations=1000, step_type=step_type,
            initialization='orthogonal', weight_distribution='uniform')
        
        # Plot results
        plt.subplot(1, 2, i+1)
        plt.semilogy(consecutive_diffs, 'b-', linewidth=2, label='||E[w_{k+1}] - E[w_k]||')
        plt.semilogy(upper_bound, 'r--', linewidth=2, label='Upper Bound (Lemma 3.2)')
        
        plt.title(f'Consecutive Mean Weight Differences ({step_type} step size)')
        plt.xlabel('Iteration k')
        plt.ylabel('||E[w_{k+1}] - E[w_k]||')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lemma_3_2_consecutive_means.png')
    plt.close()

if __name__ == "__main__":
    main() 