import numpy as np
from scipy.linalg import null_space, svdvals

def get_M2_sqrt(weight_distribution, p, n_samples, sensitivities=None):
    """
    Get the square root of M2 matrix (E[D^2]) based on the weight distribution.
    For importance sampling, M2 is diagonal with entries being the sensitivities.
    """
    if weight_distribution == 'uniform':
        val = np.sqrt(1/n_samples)
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

def get_step_sizes(n_iterations, X, M_2_sqrt, step_type='constant'):
    """
    Generate step sizes satisfying Assumption 3.1(a) and the assumption in Theorem 3.4
    
    Parameters:
    n_iterations: number of iterations
    X: data matrix
    step_type: 'constant' or 'diminishing'
    """
    X_norm = np.linalg.norm(X, ord=2)
    singular_value = get_minimum_nonzero_singular_value(X)
    norm_sigma_d = np.linalg.norm((M_2_sqrt @ M_2_sqrt) @ (M_2_sqrt @ M_2_sqrt).T, ord=2)
    c_1 = 0.9 * singular_value / (singular_value**2 + X_norm**4 * norm_sigma_d)
    
    if step_type == 'constant':
        alpha = 0.01
        return c_1 * np.ones(n_iterations + 1)
    else:
        c = 0.7 / X_norm
        return c / np.arange(1, n_iterations + 1)


def initialize_weights(X, initialization='orthogonal'):
    """
    Initialize weights satisfying Assumption 3.1(b):
    Initial guess should lie in the orthogonal complement of ker(X)
    This uses the Gram-Schmidt process to orthogonalize the initial guess.
    Returns a column vector of shape (n_features, 1)
    """
    n_samples, n_features = X.shape
    
    if initialization == 'orthogonal':
        ker_X = null_space(X)
        w = np.random.randn(n_features, 1)
        
        if ker_X.size > 0:
            for v in ker_X.T:
                v = v.reshape(-1, 1)  # Ensure column vector
                w = w - v @ (v.T @ w)
        w = w / np.linalg.norm(w)

    else:
        if initialization == 'zero':
            w = np.zeros((n_features, 1))
        else:
            w = np.random.randn(n_features, 1) * 0.01
            
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
    D = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        if weight_distribution == 'uniform':
            D[i, i] = np.random.binomial(1, 1/n_samples)
            if D[i, i] == 1:
                return D
        elif weight_distribution == 'bernoulli':
            D[i, i] = np.random.binomial(1, p)
        elif weight_distribution == 'binary':
            D[i, i] = np.random.choice([0, 1])
        elif weight_distribution == 'importance':
            if sensitivities is None:
                raise ValueError("Sensitivities must be provided for importance sampling")
            D[i, i] = np.random.binomial(1, p=sensitivities[i])
        else:
            raise ValueError(f"Unknown weight distribution: {weight_distribution}")
    
    return D

def compute_sensitivities(X, Y, w):
    """
    Compute sensitivities for importance sampling based on current residuals
    """
    residuals = Y - X @ w
    sensitivities = residuals**2
    return sensitivities / (np.sum(sensitivities) + 1e-12)  # normalize to get probabilities

