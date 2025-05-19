import numpy as np
from scipy.linalg import null_space, svdvals

def get_M2_sqrt(self, sensitivities=None):
    """
    Get the square root of M2 matrix (E[D^2]) based on the weight distribution.
    For importance sampling, M2 is diagonal with entries being the sensitivities.
    """
    if self.weight_distribution == 'uniform':
        val = np.sqrt(1/self.X.shape[0])
        return val * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'bernoulli':
        val = np.sqrt(self.p)
        return val * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'binary':
        val = np.sqrt(0.5)
        return val * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'importance':
        if sensitivities is None:
            raise ValueError("Sensitivities must be provided for importance sampling")
        # For importance sampling, M2 = diag(sensitivities) since E[D^2] = P(D=1) = sensitivities
        return np.sqrt(np.diag(sensitivities)) * np.eye(self.X.shape[0])
    else:
        raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")

def get_step_sizes(self, X, sensitivities):
    """
    Generate step sizes satisfying Assumption 3.1(a) and the assumption in Theorem 3.4
    
    Parameters:
    n_iterations: number of iterations
    X: data matrix
    step_type: 'constant' or 'diminishing'
    """
    X_norm = np.linalg.norm(X, ord=2)
    singular_value = get_minimum_nonzero_singular_value(X)
    norm_sigma_d = np.linalg.norm(get_sigma_d(self, sensitivities), ord=2)
    c_1 = 0.7 * singular_value / (singular_value**2 + X_norm**4 * norm_sigma_d)
    c = 0.7 / X_norm
    
    if self.step_type == 'constant':
        return self.alpha, self.alpha * np.ones(self.n_iterations + 1)
    elif self.step_type == 'diminishing':
        return c, c / np.arange(1, self.n_iterations + 1)
    elif self.step_type == 'diminishing_constant':
        return min(c_1, c), min(c_1, c) / np.arange(1, self.n_iterations + 1)
    else:
        raise ValueError(f"Unknown step type: {self.step_type}")


def initialize_weights(self, X):
    """
    Initialize weights satisfying Assumption 3.1(b):
    Initial guess should lie in the orthogonal complement of ker(X)
    This uses the Gram-Schmidt process to orthogonalize the initial guess.
    Returns a column vector of shape (n_features, 1)
    """
    
    if self.initialization == 'orthogonal':
        w = np.linalg.pinv(X) @ X @ self.w_true
    else:
        if self.initialization == 'zero':
            w = np.zeros((self.X.shape[1], 1))
        else:
            w = np.random.randn(self.X.shape[1], 1) * 0.01
            
    return w

def get_minimum_nonzero_singular_value(X):
    """
    Compute the minimum non-zero singular value of X
    """
    singular_values = svdvals(X)
    nonzero_singular_values = singular_values[singular_values > 1e-10]
    return np.min(nonzero_singular_values) if len(nonzero_singular_values) > 0 else 0

def compute_S_alpha(self, A, X, Y, w_hat, alpha, M2_sqrt, sensitivities):
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
    
    sigma_D = get_sigma_d(self, sensitivities)
    hadamard_term = sigma_D * (XAXt + residual_outer)
    
    # Complete second term
    term2 = (alpha**2) * X.T @ hadamard_term @ X
    
    return term1 + term2

def get_D_matrix(self, sensitivities=None):
    """
    Generate diagonal matrix D based on the specified distribution
    """
    D = np.zeros((self.X.shape[0], self.X.shape[0]))
    
    for i in range(self.X.shape[0]):
        if self.weight_distribution == 'uniform':
            D[i, i] = np.random.binomial(1, 1/self.X.shape[0])
        elif self.weight_distribution == 'bernoulli':
            D[i, i] = np.random.binomial(1, self.p)
        elif self.weight_distribution == 'binary':
            D[i, i] =  np.random.binomial(1, 0.5)
        elif self.weight_distribution == 'importance':
            if sensitivities is None:
                raise ValueError("Sensitivities must be provided for importance sampling")
            D[i, i] = np.random.binomial(1, p=sensitivities[i])
        else:
            raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")
    
    return D

def compute_sensitivities(self, w):
    """
    Compute sensitivities for importance sampling based on current residuals
    """
    residuals = self.Y - self.X @ w
    sensitivities = residuals**2
    return sensitivities / (np.sum(sensitivities) + 1e-12)  # normalize to get probabilities

def get_sigma_d(self, sensitivities):
    """
    Compute the norm of sigma_d
    """

    if self.weight_distribution == 'bernoulli':
        return self.p * (1-self.p) * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'uniform':
        p = 1/self.X.shape[0]
        return p * (1-p) * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'binary':
        p = 0.5
        return p * (1-p) * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'importance':
        sigma_sensitivity = np.zeros((sensitivities.shape[0]))
        for i in range(sensitivities.shape[0]):
            sigma_sensitivity[i] = sensitivities[i] * (1 - sensitivities[i])
        return sigma_sensitivity
    else:
        raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")