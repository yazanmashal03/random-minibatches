import matplotlib.pyplot as plt
from util.helper import *
from scipy.special import zeta

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
        sensitivities = None
        if self.weight_distribution == 'importance':
            s_init = np.random.randn(self.X.shape[1], 1)
            sensitivities = compute_sensitivities(self, s_init)
        
        # Used to compute the true solution and satisfy the assumptions of the theorems.
        M_2_sqrt = get_M2_sqrt(self, sensitivities)
        X_hat = M_2_sqrt @ self.X
        Y_hat = M_2_sqrt @ self.Y
        w_hat = np.linalg.pinv(X_hat) @ Y_hat
        alphab, alphas = get_step_sizes(self, self.X.T @ (M_2_sqrt @ M_2_sqrt) @ self.X, sensitivities)

        # Initialize weights
        w_init = initialize_weights(self, X_hat)

        # Store results from multiple simulations
        all_diffs = np.zeros((self.n_simulations, self.n_iterations, self.X.shape[1], 1))
        all_second_diffs = np.zeros((self.n_simulations, self.n_iterations, self.X.shape[1], self.X.shape[1]))

        # Run multiple simulations
        for sim in range(self.n_simulations):
            w = w_init.copy()
            
            for k in range(self.n_iterations):
                sensitivities = None
                if self.weight_distribution == 'importance':
                    sensitivities = compute_sensitivities(self, w)
                
                D_k = get_D_matrix(self, sensitivities)
                
                # Update weights
                D_k_squared = D_k @ D_k

                I = np.eye(self.X.shape[1])
                # This is the update rule for the weights from equation (3) in the paper
                w = (I - alphas[k] * self.X.T @ D_k_squared @ self.X) @ w + alphas[k] * self.X.T @ D_k_squared @ self.Y
                all_diffs[sim, k] = w - w_hat
                all_second_diffs[sim, k] = (w - w_hat) @ (w - w_hat).T
            print(sim)

        # Compute first moment (mean)
        mean_weights = np.mean(all_diffs, axis=0)
        first_moment = np.linalg.norm(mean_weights, axis=(1,2), ord=2)
        
        all_second_means = np.mean(all_second_diffs, axis=0)
        second_moment = np.linalg.norm(all_second_means, axis=(1,2), ord=2)
        second_moment_diff = np.zeros(self.n_iterations)
        norm_sigma_d = np.linalg.norm(get_sigma_d(self, sensitivities), ord=2)

        for k in range(self.n_iterations):
            second_moment_diff[k] = np.linalg.norm(all_second_means[k] - compute_S_alpha(self, all_second_means[k], self.X, self.Y, w_hat, alphas[k], M_2_sqrt, sensitivities), ord=2)

        # Compute theoretical bounds
        X_norm = np.linalg.norm(self.X, ord=2)
        sigma_min = get_minimum_nonzero_singular_value(self.X.T @ (M_2_sqrt @ M_2_sqrt) @ self.X)
        euler_gamma = 0.5772156649

        first_moment_bound = np.zeros(self.n_iterations)
        second_moment_bound = np.zeros(self.n_iterations)
        second_moment_diff_bound = np.zeros(self.n_iterations)

        # defining the constants for the bounds
        C_0 = np.linalg.norm((w_init - w_hat) @ (w_init - w_hat).T, ord=2) + 2 * X_norm**3 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2) * np.linalg.norm(w_init - w_hat, ord=2)
        C_1 = C_0 * (1 + alphab * ((np.pi)**2/6)) + X_norm**2 * norm_sigma_d * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)**2 * (np.e)**(alphab * sigma_min * euler_gamma) * alphab * zeta(2-alphab * sigma_min)   
        
        # Compute the product term for each k
        for k in range(self.n_iterations):

            product = 1.0
            for l in range(k):
                product *= (1 - alphas[l] * sigma_min)
            first_moment_bound[k] = product * np.linalg.norm(w_init - w_hat, ord=2)
            second_moment_diff_bound[k] = 2 * alphas[k]**2 * X_norm**3 * norm_sigma_d * product * np.linalg.norm(w_init - w_hat, ord=2) * np.linalg.norm(self.Y - self.X @ w_hat, ord=2)
            second_moment_bound[k] = C_1 * (1/((k)**(alphab * sigma_min)))
        
        return first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound

def main():
    # Generate synthetic data
    np.random.seed(31)
    n_samples = 10
    n_features = 100

    n_iterations = 500
    n_simulations = 10
    p_succ = 0.2
    weight_distribution = 'bernoulli'
    initialization = 'orthogonal'
    alpha = 0.01 # this is the step size for the constant step size
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
    for i in range(n_samples):
        sigma[i,i] = np.random.binomial(1, p=0.5) * np.random.randn()

    X = Q @ sigma @ S
    print("This is I - XX^+: ", np.linalg.norm(np.eye(n_samples) - X @ np.linalg.pinv(X), ord=2))
    
    # Generate true weights (sparse) I am assuming that the true weights are sparse, since we are working with an over-parameterized model
    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    wg = WeightedGD(X, Y, w_true, alpha, n_iterations, step_types[0], initialization,
                    weight_distribution, p_succ, n_simulations)
    
    for i, step_type in enumerate(step_types):
        first_moment, first_moment_bound, second_moment_diff, second_moment_diff_bound, second_moment, second_moment_bound = wg.simulate_weighted_gd()
        
        # Plot the convergence of the iterates to the expected solution
        # --- First plot: First moment convergence ---
        plt.figure(figsize=(8, 5))
        plt.plot(first_moment, 'b-', linewidth=2, label=r'$\|\mathbb{E}_D[\hat w_{k+1} - \hat{w}]\|_2$')
        plt.plot(first_moment_bound, 'r--', linewidth=2, label='First moment bound (Lemma 3.2)')
        plt.title(f'First Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('First moment')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Second plot: Second moment vs operator S(.) ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment_diff, 'b-', linewidth=2,
                label=r'$\|\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T] - S_{\alpha_k}(\mathbb{E}_D[(\hat w_k - \hat{w})(Y - X \hat{w})^T]) \|_2$')
        plt.plot(second_moment_diff_bound, 'r--', linewidth=2, label='Second moment bound (Lemma 3.3)')
        plt.title(f'Second Moment Convergence ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment difference')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Third plot: Norm of second moment ---
        plt.figure(figsize=(8, 5))
        plt.plot(second_moment, 'b-', linewidth=2,
                label = r'$\|\mathbb{E}_{D}[(\hat{w}_{k+1} - \hat{w})(\hat{w}_{k+1} - \hat{w})^\top]\|_2$')
        #plt.plot(second_moment_bound, 'r--', linewidth=2, label='Second moment bound (Lemma 3.3)')
        plt.title(f'Norm of the Second Moment ({step_type} step size)')
        plt.xlabel('Iteration $k$')
        plt.ylabel('Second moment')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 