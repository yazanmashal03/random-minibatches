import matplotlib.pyplot as plt
from util.helper import *
from scipy.special import zeta
from scipy.linalg import sqrtm

class WeightedGD:
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
            ValueError("M_2 is likely singular or ill-conditioned.")

        M_2_sqrt = psd_sqrt(M_2)
        print(np.allclose(M_2, M_2_sqrt @ M_2_sqrt))  # This will print True if correct
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
            print("This is simulation: ", sim)

        # Compute first moment (mean)
        all_diffs = all_diffs.squeeze()
        mean_weights = np.mean(all_diffs, axis=0)
        first_moment = np.linalg.norm(mean_weights, axis=1, ord=2)
        
        all_second_means = np.mean(all_second_diffs, axis=0)
        second_moment = np.linalg.norm(all_second_means, axis=(1,2), ord=2)
        second_moment_diff = np.zeros(self.n_iterations)
        norm_sigma_d = np.linalg.norm(get_sigma_d(self), ord=2)

        for k in range(self.n_iterations):
            second_moment_diff[k] = np.linalg.norm(all_second_means[k] - compute_S_alpha(self, all_second_means[k], self.X, self.Y, w_hat, alphas[k], M_2_sqrt, self.sensitivities), ord=2)

        # Compute theoretical bounds
        first_moment_bound = np.zeros(self.n_iterations)
        second_moment_bound = np.zeros(self.n_iterations)
        second_moment_diff_bound = np.zeros(self.n_iterations)

        # defining the constants for the bounds
        C_0 = np.linalg.norm((w_init - w_hat) @ (w_init - w_hat).T, ord=2) + 2 * X_norm**3 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2) * np.linalg.norm(w_init - w_hat, ord=2)
        C_1 = C_0 * (1 + alphab * (((np.pi)**2)/6)) + X_norm**2 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)**2 * (np.e)**(alphab * sigma_min * euler_gamma) * alphab * zeta(2 - (alphab * sigma_min))
        #C_1 = 22
        print("This is C_1: ", C_1)
        print("This is alpha: ", alphab, "This is sigma_min: ", sigma_min)
        
        # Compute the product term for each k
        for k in range(self.n_iterations):

            product = 1.0
            for l in range(k):
                product *= (1 - alphas[l] * sigma_min)
            first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
            second_moment_diff_bound[k] = 2 * alphas[k]**2 * X_norm**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)
            second_moment_bound[k] = C_1 * (1/((k+1)**(alphab * sigma_min)))
        
        return first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(33)
    n_samples = 5
    n_features = 50

    n_iterations = 1000
    n_simulations = 100
    p_succ = 0.2
    weight_distribution = 'uniform'
    initialization = 'orthogonal'
    alpha = 0.65 # this is the step size for the constant step size
    step_types = ['constant']
    
    # Generate random data matrix X
    # X = np.random.randn(n_samples, n_features)
    # print("This is I - XX^+: ", np.linalg.norm(np.eye(n_samples) - X @ np.linalg.pinv(X), ord=2))

    G = np.random.randn(n_samples, n_samples)
    # do a QR decomposition
    Q, R = np.linalg.qr(G)

    A = np.random.randn(n_features, n_features)
    S,T = np.linalg.qr(A)

    # Generate Cauchy distributed values and scale them down
    sigma = np.zeros((n_samples, n_features))

    # Set diagonal elements to zero with probability p
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p=0.6) * np.abs(np.random.randn())

    X = Q @ sigma @ S
    print("This is I - XX^+: ", np.linalg.norm(np.eye(n_samples) - X @ np.linalg.pinv(X), ord=2))
    
    # Generate true weights
    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    wg = WeightedGD(X, Y, w_true, alpha, n_iterations, step_types[0], initialization,
                    weight_distribution, p_succ, n_simulations)
    
    for i, step_type in enumerate(step_types):
        first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound = wg.simulate_weighted_gd()
        
        # normalizing the moments, and the bounds
        # first_moment = first_moment / (first_moment.max() + 1e-10)
        # first_moment_bound = first_moment_bound / first_moment_bound.max() + 1e-10
        # second_moment_diff = second_moment_diff / second_moment_diff.max() + 1e-10
        # second_moment_diff_bound = second_moment_diff_bound / second_moment_diff_bound.max() + 1e-10
        # second_moment = second_moment / (second_moment.max() + 1e-10)
        # second_moment_bound = second_moment_bound / (second_moment_bound.max() + 1e-10)
        
        # Plot the convergence of the iterates to the expected solution
        # --- First plot: First moment convergence ---
        plt.figure(figsize=(8, 5))
        plt.plot(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[\hat w_{k+1} - \hat{w}]\|_2$')
        plt.plot(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        plt.plot(np.abs(first_moment - first_moment_bound), 'g--', linewidth=2, label='Covergence of the bound')
        plt.title(f'First Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('First moment')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Second plot: Second moment vs operator S(.) ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment_diff, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        plt.plot(second_moment_diff_bound, 'r--', linewidth=2, label='Second moment minus S(.) bound (Lemma 3.3)')
        plt.plot(np.abs(second_moment_diff - second_moment_diff_bound), 'g--', linewidth=2, label='Covergence of the bound')
        plt.title(f'Second Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment minus S(.)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Third plot: Norm of second moment ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment, 'b-', linewidth=2, label = r'$\|\mathbb{E}_{D}[(\hat{w}_{k+1} - \hat{w})(\hat{w}_{k+1} - \hat{w})^\top]\|_2$')
        plt.plot(second_moment_bound, 'r--', linewidth=2, label='Second moment bound (Theorem 3.4)')
        plt.plot(np.abs(second_moment - second_moment_bound), 'g--', linewidth=2, label='Covergence of the bound')
        plt.title(f'Norm of the Second Moment ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 