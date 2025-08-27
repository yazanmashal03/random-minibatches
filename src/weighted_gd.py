import numpy as np
from utils.helper import *
from scipy.special import zeta

class WeightedGD:
    """
    Class to simulate weighted gradient descent and compute moments
    """
    def __init__(self, X, Y, alpha, n_iterations, step_type, initialization, weight_distribution, weights, batch_size, n_simulations):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.step_type = step_type
        self.initialization = initialization
        self.weight_distribution = weight_distribution
        self.weights = weights
        self.batch_size = batch_size
        self.n_simulations = n_simulations


    def simulate_weighted_gd(self):
        """
        Simulate randomly weighted gradient descent and compute moments
        """

        # Normalize weights into probabilities
        self.probabilities = np.exp(self.weights) / np.sum(np.exp(self.weights))

        # Second moment of noise
        M_2 = get_M2(self.probabilities, self.weight_distribution, self.batch_size)

        # Check condition number
        cond_number = np.linalg.cond(M_2)

        if np.isfinite(cond_number):

            print("M_2 is non-singular.")

        else:

            ValueError("M_2 is likely singular or ill-conditioned.")

        # Pre-compute some objects
        M_2_sqrt = np.sqrt(M_2)
        sigma_D = get_sigma_d(self.probabilities, self.weight_distribution, self.batch_size)
        norm_sigma_d = np.linalg.norm(sigma_D, ord=2)

        X_hat = M_2_sqrt @ self.X
        X_hat_sq = X_hat.T @ X_hat
        Y_hat = M_2_sqrt @ self.Y
        w_hat = np.linalg.pinv(X_hat) @ Y_hat

        inter_term = self.Y - self.X @ w_hat
        I_d = np.eye(self.X.shape[1])
        exp_aff_shift = self.X.T @ (sigma_D * (inter_term @ inter_term.T)) @ self.X

        X_norm = np.linalg.norm(self.X, ord=2)
        sv_hat = np.linalg.svdvals(X_hat_sq)
        sigma_min = np.min(sv_hat[np.greater(sv_hat, 1e-8)])
        euler_gamma = 0.5772156649

        # Generate step-sizes
        neu_cut = 100

        if self.step_type == 'constant':

            alphas = self.alpha * np.ones(np.max((self.n_iterations, neu_cut)))

        elif self.step_type == 'linear':

            alphas = self.alpha / np.arange(1, np.max((self.n_iterations, neu_cut)))

        else:

            raise ValueError(f"Unknown step type: {self.step_type}")

        # Estimate asymptotic variance via Neumann series
        lim_var = np.zeros(X_hat_sq.shape)

        for m in range(neu_cut):
            lim_var = compute_S_alpha(lim_var, alphas[m], self.X, X_hat_sq, sigma_D, exp_aff_shift)

        # Initialize weights
        w_init = initialize_weights(self.initialization, self.X)

        # Store averages
        mean_diffs = np.zeros((self.n_iterations, self.X.shape[1]))
        square_dist = np.zeros(self.n_iterations)
        square_diffs = np.zeros((self.n_iterations, self.X.shape[1], self.X.shape[1]))

        # Run multiple simulations
        for sim in range(self.n_simulations):

            w_diff = w_init.copy() - w_hat

            for k in range(self.n_iterations):

                # Sample random weights
                D_k = get_D_matrix(self.probabilities, self.weight_distribution, self.batch_size)
                D_k_squared = D_k @ D_k

                # This is the main update rule
                w_diff = (I_d - alphas[k] * self.X.T @ D_k_squared @ self.X) @ w_diff + alphas[k] * self.X.T @ D_k_squared @ inter_term

                # Store first and second-order differences
                mean_diffs[k] += w_diff.squeeze() / self.n_simulations
                square_dist[k] += np.linalg.norm(w_diff)**2 / self.n_simulations
                square_diffs[k] += ((w_diff) @ (w_diff).T - lim_var) / self.n_simulations

            print("Just finished simulation: ", sim + 1)

        # Compute moments
        try:

            # Compute mean convergence
            first_moment = np.linalg.norm(mean_diffs, axis=1)

            # Compute second moment convergence
            second_moment = np.linalg.norm(square_diffs, axis=(1, 2), ord=2)

        except Exception as e:

            print(f"Error converging: {e}")
            return None, None, None, None, None, None

        # Compute theoretical bounds
        first_moment_bound = np.zeros(self.n_iterations)
        second_moment_bound = np.zeros(self.n_iterations)
        # second_moment_diff_bound = np.zeros(self.n_iterations)

        # Constants for the bounds
        aff_norm = X_norm**3 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat) * np.linalg.norm(w_init - w_hat)
        C_0 = np.linalg.norm((w_init - w_hat) @ (w_init - w_hat).T, ord=2) + 2 * self.alpha**2 * aff_norm

        if self.step_type == 'constant':

            C_2 = C_0 + np.linalg.norm(lim_var, ord=2)

        elif self.step_type == 'linear':

            C_1 = C_0 * (1 + self.alpha * (((np.pi)**2)/6)) + X_norm**2 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat)**2 * np.exp(alpha * sigma_min * euler_gamma) * alpha * zeta(2 - alpha * sigma_min)

        else:

            raise ValueError(f"Unknown step type: {self.step_type}")

        # Recursively compute the bounds
        product = 1.0

        for k in range(self.n_iterations):

            product *= (1 - alphas[k] * sigma_min)

            first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat)

            if self.step_type == 'constant':
                second_moment_bound[k] = C_2 * (2 + (k + 1) * self.alpha**2) * product
            elif self.step_type == 'linear':
                second_moment_bound[k] = C_1 * product
            else:
                raise ValueError(f"Unknown step type: {self.step_type}")

        return first_moment, first_moment_bound, second_moment, second_moment_bound, square_dist
