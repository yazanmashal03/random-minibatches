import numpy as np
from utils.helper import *
from scipy.special import zeta

class WeightedGD:
    """
    Class to simulate weighted gradient descent and compute moments
    """
    def __init__(self, X, Y, w_true, alpha, n_iterations, step_type, initialization,
                        weight_distribution, p, n_simulations):
        self.X = X
        self.Y = Y
        self.w_true = w_true
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.step_type = step_type
        self.initialization = initialization
        self.weight_distribution = weight_distribution
        self.p = p
        self.n_simulations = n_simulations
        self.sensitivities = None

    def simulate_weighted_gd(self):
        """
        Simulate randomly weighted gradient descent and compute moments
        """

        # Compute initial sensitivities if using importance sampling
        if self.weight_distribution == 'importance':
            sensitivities = compute_sensitivities(self)
            self.sensitivities = sensitivities
        
        # Used to compute the true solution and satisfy the assumptions of the theorems.
        M_2 = get_M2(self)

        # Check if the matrix is square
        if M_2.shape[0] != M_2.shape[1]:
            raise ValueError("Matrix is not square!")

        # Check condition number
        cond_number = np.linalg.cond(M_2)
        if cond_number < 1/np.finfo(M_2.dtype).eps:
            print("M_2 is non-singular (well-conditioned).")
        else:
            raise ValueError("M_2 is likely singular or ill-conditioned.")

        M_2_sqrt = psd_sqrt(M_2)
        X_hat = M_2_sqrt @ self.X
        Y_hat = M_2_sqrt @ self.Y
        w_hat = np.linalg.pinv(X_hat) @ Y_hat
        alphab, alphas = get_step_sizes(self, self.X.T @ (M_2) @ self.X)

        X_norm = np.linalg.norm(self.X, ord=2)
        sigma_min = get_minimum_nonzero_singular_value(self.X.T @ (M_2) @ self.X)
        euler_gamma = 0.5772156649

        # Initialize weights
        w_init = initialize_weights(self, X_hat)

        # Store results from multiple simulations
        all_diffs = np.zeros((self.n_simulations, self.n_iterations, self.X.shape[1], 1))
        all_second_diffs = np.zeros((self.n_simulations, self.n_iterations, self.X.shape[1], self.X.shape[1]))

        # Run multiple simulations
        for sim in range(self.n_simulations):
            w_diff = w_init.copy() - w_hat
            
            for k in range(self.n_iterations):
                D_k = get_D_matrix(self)

                # Update weights
                D_k_squared = D_k @ D_k

                I = np.eye(self.X.shape[1])
                # This is the main update rule
                w_diff = (I - alphas[k] * self.X.T @ D_k_squared @ self.X) @ w_diff + alphas[k] * self.X.T @ D_k_squared @ (self.Y-self.X @ w_hat)
                all_diffs[sim, k] = w_diff
                all_second_diffs[sim, k] = (w_diff) @ (w_diff).T
            print("Just finished simulation: ", sim)

        # Compute first moment (mean)
        try:
            all_diffs = all_diffs.squeeze()
            mean_weights = np.mean(all_diffs, axis=0)
            first_moment = np.linalg.norm(mean_weights, axis=1, ord=2)
            all_second_means = np.mean(all_second_diffs, axis=0)
            second_moment = np.linalg.norm(all_second_means, axis=(1,2), ord=2) # this is the second moment of the weights
        except Exception as e:
            print(f"Error converging: {e}")
            return None, None, None, None, None, None
        
        second_moment_diff = np.zeros(self.n_iterations)
        norm_sigma_d = np.linalg.norm(get_sigma_d(self), ord=2)

        for k in range(self.n_iterations):
            second_moment_diff[k] = np.linalg.norm(all_second_means[k] - compute_S_alpha(self, all_second_means[k], self.X, self.Y, w_hat, alphas[k], M_2_sqrt), ord=2)

        # Compute theoretical bounds
        first_moment_bound = np.zeros(self.n_iterations)
        second_moment_bound = np.zeros(self.n_iterations)
        second_moment_diff_bound = np.zeros(self.n_iterations)

        # defining the constants for the bounds
        C_0 = np.linalg.norm((w_init - w_hat) @ (w_init - w_hat).T, ord=2) + 2 * X_norm**3 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2) * np.linalg.norm(w_init - w_hat, ord=2)
        C_1 = C_0 * (1 + alphab * (((np.pi)**2)/6)) + X_norm**2 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)**2 * (np.e)**(alphab * sigma_min * euler_gamma) * alphab * zeta(2 - (alphab * sigma_min))
        print("This is C_1: ", C_1)
        print("This is alpha: ", alphab, "This is sigma_min: ", sigma_min, "This is alphab * sigma_min: ", alphab * sigma_min)
        
        # Compute the product term for each k
        for k in range(self.n_iterations):

            product = 1.0
            for l in range(k):
                product *= (1 - alphas[l] * sigma_min)
            first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
            second_moment_diff_bound[k] = 2 * alphas[k]**2 * X_norm**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)
            second_moment_bound[k] = C_1 * (1/((k+1)**(alphab * sigma_min)))
        
        return first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound
