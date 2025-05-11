import numpy as np
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
        alpha = 0.001
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
    return sensitivities / (np.sum(sensitivities) + 1e-12)  # normalize to get probabilities

