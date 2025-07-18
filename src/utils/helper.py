import numpy as np
from scipy.linalg import svdvals
import matplotlib.pyplot as plt
import os

def get_M2(self):
    """
    Get the expected M2 matrix (E[D^2]) based on the weight distribution.
    For importance sampling, M2 is diagonal with entries being the sensitivities.
    """
    if self.weight_distribution == 'uniform':
        val = (1/self.X.shape[0])
        return val * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'bernoulli':
        return self.p * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'binary':
        val = (0.5)
        return val * np.eye(self.X.shape[0])
    elif self.weight_distribution == 'importance':
        if self.sensitivities is None:
            raise ValueError("Sensitivities must be provided for importance sampling")
        return np.diag(self.sensitivities)
    elif self.weight_distribution == 'noiseless':
        return np.eye(self.X.shape[0])
    else:
        raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")

def get_step_sizes(self, X):
    """
    Generate step sizes satisfying Assumption 3.1(a) and the assumption in Theorem 3.4
    
    Parameters:
    n_iterations: number of iterations
    X: data matrix
    step_type: 'constant' or 'diminishing'
    """
    X_norm = np.linalg.norm(X, ord=2)
    singular_value = get_minimum_nonzero_singular_value(X)
    norm_sigma_d = np.linalg.norm(get_sigma_d(self), ord=2)
    c_1 = 0.7 * singular_value / (singular_value**2 + X_norm**4 * norm_sigma_d)
    c = 0.9 / X_norm
    
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
    elif self.initialization == 'zero':
            w = np.zeros((self.X.shape[1], 1))
    elif self.initialization == 'ones':
            w = np.ones((self.X.shape[1], 1))
    else:
        w = np.random.randn(self.X.shape[1], 1)
            
    return w

def get_minimum_nonzero_singular_value(X):
    """
    Compute the minimum non-zero singular value of X
    """
    singular_values = svdvals(X)
    nonzero_singular_values = singular_values[singular_values > 1e-10]
    return np.min(nonzero_singular_values) if len(nonzero_singular_values) > 0 else 0

def compute_S_alpha(self, A, X, Y, w_hat, alpha, M2_sqrt):
    """
    Compute the S_alpha operator from Lemma 3.3:
    S_alpha(A) = (I - alpha X_hat)A(I - alpha X_hat) + alpha^2 X^T(Sigma_D ○ (XAX^T + (Y-Xw_hat)(Y-Xw_hat)^T))X
    """
    n_samples, n_features = X.shape
    I = np.eye(n_features)
    
    # First term: (I - alpha X_hat)A(I - alpha X_hat)
    term1 = (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X) @ A @ (I - alpha * X.T @ (M2_sqrt @ M2_sqrt) @ X)
    
    # Second term components
    XAXt = X @ A @ X.T
    residual_outer = (Y - X @ w_hat) @ (Y - X @ w_hat).T
    
    sigma_D = get_sigma_d(self)
    hadamard_term = sigma_D * (XAXt + residual_outer)
    
    # Complete second term
    term2 = (alpha**2) * X.T @ hadamard_term @ X
    
    return term1 + term2

def get_D_matrix(self):
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
            if self.sensitivities is None:
                raise ValueError("Sensitivities must be provided for importance sampling")
            D[i, i] = self.sensitivities[i]
        elif self.weight_distribution == 'noiseless':
            D = np.eye(self.X.shape[0])
        else:
            raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")
    
    return D

def compute_sensitivities(self):
    """
    Compute sensitivities for importance sampling based on row norms of X.
    Rows with larger L2 norms will have higher sensitivity values.
    """
    # Compute L2 norm of each row in X
    row_norms = np.linalg.norm(self.X, axis=1)
    sensitivities = row_norms**2
    return sensitivities / (np.sum(sensitivities) + 1e-12)  # normalize to get probabilities

def get_sigma_d(self):
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
        sigma_sensitivity = self.sensitivities * (1 - self.sensitivities)
        return np.diag(sigma_sensitivity)
    elif self.weight_distribution == 'noiseless':
        return np.eye(self.X.shape[0])
    else:
        raise ValueError(f"Unknown weight distribution: {self.weight_distribution}")
    
def psd_sqrt(M):
    # Ensure M is symmetric
    M = (M + M.T) / 2  

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(M)

    # Clip small negative eigenvalues due to numerical error
    eigvals_clipped = np.clip(eigvals, 0, None)

    # Compute the square root
    sqrt_M = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T

    return sqrt_M

def plot_simulation_results(first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound, step_type, save_path=None):
    """
    Plot the simulation results
    """
    if first_moment is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "Error converging", fontsize=16, ha='center', va='center')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        return
    try:
        # Create a single figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        print("Figure created successfully")
        
        # First subplot: First moment convergence
        ax1.plot(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[\hat w_{k+1} - \hat{w}]\|_2$')
        ax1.plot(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        ax1.plot(np.abs(first_moment - first_moment_bound), 'g--', linewidth=2, label='Covergence of the bound')
        ax1.set_title(f'First Moment Convergence ({step_type} step size)')
        ax1.set_xlabel('Iteration $k$')
        ax1.set_ylabel('First moment')
        ax1.legend(loc='best')
        ax1.grid(True)
        print("First subplot created")

        # Second subplot: Second moment vs operator S(.)
        ax2.plot(second_moment_diff, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        ax2.plot(second_moment_diff_bound, 'r--', linewidth=2, label='Second moment minus S(.) bound (Lemma 3.3)')
        ax2.plot(np.abs(second_moment_diff - second_moment_diff_bound), 'g--', linewidth=2, label='Covergence of the bound')
        ax2.set_title(f'Second Moment Convergence ({step_type} step size)')
        ax2.set_xlabel('Iteration $k$')
        ax2.set_ylabel('Second moment minus S(.)')
        ax2.legend(loc='best')
        ax2.grid(True)
        print("Second subplot created")

        # Third subplot: Norm of second moment
        ax3.plot(second_moment, 'b-', linewidth=2, label = r'$\|\mathbb{E}_{D}[(\hat{w}_{k+1} - \hat{w})(\hat{w}_{k+1} - \hat{w})^\top]\|_2$')
        ax3.plot(second_moment_bound, 'r--', linewidth=2, label='Second moment bound (Theorem 3.4)')
        ax3.plot(np.abs(second_moment - second_moment_bound), 'g--', linewidth=2, label='Covergence of the bound')
        ax3.set_title(f'Norm of the Second Moment ({step_type} step size)')
        ax3.set_xlabel('Iteration $k$')
        ax3.set_ylabel('Second moment')
        ax3.legend(loc='best')
        ax3.grid(True)
        print("Third subplot created")
        
        plt.tight_layout()
        print("Layout adjusted")
        
        if save_path is not None:
            print(f"Attempting to save figure to: {save_path}")
            print(f"Figure size: {fig.get_size_inches()}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Try to save with explicit parameters
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print("plt.savefig() completed")
            
            plt.close(fig)
            print("Figure closed")
            
            # Verify the file was created
            if os.path.exists(save_path):
                print(f"✓ File successfully saved: {save_path}")
                print(f"  File size: {os.path.getsize(save_path)} bytes")
            else:
                print(f"✗ File not found after save: {save_path}")
                print(f"  Directory contents: {os.listdir(os.path.dirname(save_path))}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error in plot_simulation_results: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)
        raise  # Re-raise the exception so it's not silently caught
