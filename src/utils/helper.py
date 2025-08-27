import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import os


def get_M2(probabilities, weight_distribution, batch_size):
    """
    Get the expected weighting matrix based on the weight distribution.
    """

    if weight_distribution == 'bernoulli':

        return np.diag(probabilities)

    elif weight_distribution == 'noiseless':

        return np.eye(probabilities.shape[0])

    else:

        raise ValueError(f"Unknown weight distribution: {weight_distribution}")


def initialize_weights(initialization, X):
    """
    Initialize gradient descent parameter
    """

    if initialization == 'orthogonal': # Initialize orthogonal to the kernel of X

        w = 5 * np.linalg.pinv(X) @ X @ np.random.randn(X.shape[1], 1)

    elif initialization == 'zero':

        w = np.zeros((X.shape[1], 1))

    elif initialization == 'ones':

        w = np.ones((X.shape[1], 1))

    else:

        w = np.random.randn(X.shape[1], 1)

    return w


def compute_S_alpha(A, alpha, X, X_hat_sq, sigma_D, exp_aff_shift):
    """
    Compute the affine operator from Lemma 3.3
    """

    I_d = np.eye(X_hat_sq.shape[0])

    # First linear term
    lin_term_1 = (I_d - alpha * X_hat_sq) @ A @ (I_d - alpha * X_hat_sq)

    # Second linear term
    lin_term_2 = X.T @ (sigma_D * (X @ A @ X.T)) @ X

    return lin_term_1 + alpha**2 * (lin_term_2 + exp_aff_shift)


def get_D_matrix(probabilities, weight_distribution, batch_size):
    """
    Generate weighting matrix based on the specified distribution
    """

    if weight_distribution == 'bernoulli':

        batch = rng.choice(probabilities.shape[0], size = batch_size, replace = False, p = probabilities)
        D_diag = np.zeros(probabilities.shape[0])
        D_diag[batch] = 1
        D = np.diag(D_diag)

    elif weight_distribution == 'noiseless':

        D = np.eye(probabilities.shape[0])

    else:

        raise ValueError(f"Unknown weight distribution: {weight_distribution}")

    return D


def get_sigma_d(probabilities, weight_distribution, batch_size):
    """
    Compute covariance of squared weighting matrix
    """
    if batch_size != 'random' and batch_size > 1:

        raise ValueError(f"Only mini-batch sizes 1 and 'random' are implemented so far!")

    elif weight_distribution == 'noiseless':

        return np.eye(probabilities.shape[0])

    elif batch_size == 'random' and weight_distribution == 'bernoulli':

        return np.diag(probabilities * (1 - probabilities))

    elif batch_size == 1 and weight_distribution == 'bernoulli':

        return np.diag(probabilities) - np.outer(probabilities, probabilities)

    else:

        raise ValueError(f"Unknown weight distribution: {weight_distribution}")


def plot_simulation_results(first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist, step_type, save_path=None):
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
        ax1.plot(first_moment, 'b-', linewidth=2, label='Empirical')
        ax1.plot(first_moment_bound, 'r--', linewidth=2, label='Lemma 3.2')
        ax1.set_title(f'First Moment Convergence ({step_type} step size)')
        ax1.set_xlabel('Iteration $k$')
        ax1.set_ylabel('First moment')
        ax1.legend(loc='best')
        ax1.grid(True)
        print("First subplot created")

        # Second subplot: Second moment vs operator S(.)
        # ax2.plot(second_moment_diff, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        # ax2.plot(second_moment_diff_bound, 'r--', linewidth=2, label='Second moment minus S(.) bound (Lemma 3.3)')
        # ax2.set_title(f'Second Moment Convergence ({step_type} step size)')
        # ax2.set_xlabel('Iteration $k$')
        # ax2.set_ylabel('Second moment minus S(.)')
        # ax2.legend(loc='best')
        # ax2.grid(True)
        # print("Second subplot created")

        # Second subplot: Second moment convergence
        ax2.plot(second_moment, 'b-', linewidth=2, label = 'Empirical')
        ax2.plot(second_moment_bound, 'r--', linewidth=2, label='Theorem 3.4')
        ax2.set_title(f'Second Moment Convergence ({step_type} step size)')
        ax2.set_xlabel('Iteration $k$')
        ax2.set_ylabel('Second moment')
        ax2.legend(loc='best')
        ax2.grid(True)
        print("Second subplot created")

        # Third subplot: Convergence in squared distance
        ax3.plot(square_dist, 'b-', linewidth=2)
        ax3.set_title(f'Convergence in Squared Distance ({step_type} step size)')
        ax3.set_xlabel('Iteration $k$')
        ax3.set_ylabel('Squared distance')
        # ax3.legend(loc='best')
        ax3.grid(True)
        print("Second subplot created")

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
