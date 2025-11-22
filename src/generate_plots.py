import numpy as np
import matplotlib.pyplot as plt
from weighted_gd import WeightedGD

def run_comparison():
    # --- 1. Data Generation ---
    np.random.seed(40)
    n_samples = 100
    n_features = 10
    n_iterations = 500
    n_simulations = 100
    alpha = 0.25
    batch_size = 1
    
    # Data generation (same as in your main)
    G = np.random.randn(n_samples, n_samples)
    Q, R = np.linalg.qr(G)
    A = np.random.randn(n_features, n_features)
    S, T = np.linalg.qr(A)
    sigma = np.zeros((n_samples, n_features))
    for i in range(min(n_samples, n_features)):
        sigma[i,i] = np.random.binomial(1, p=0.4) * np.abs(1 + np.random.randn())
    X = Q @ sigma @ S

    row_scalers = np.ones(n_samples)
    num_large = int(0.1 * n_samples)
    row_scalers[:num_large] = 10.0  # "Important" data points
    row_scalers[num_large:] = 0.5   # "Unimportant" data points
    X = X * row_scalers[:, np.newaxis]

    w_true = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1)
    Y = X @ w_true + noise

    # --- 2. Run Simulation: Uniform Sampling ---
    weights_uniform = np.zeros(n_samples) 
    
    gd_uniform = WeightedGD(
        X, Y, alpha, n_iterations, 'constant', 'orthogonal', 
        'bernoulli', weights_uniform, batch_size, n_simulations
    )
    _, _, _, _, dist_uniform = gd_uniform.simulate_weighted_gd()

    # --- 3. Run Simulation: Importance Sampling ---
    print("Running Importance Sampling...")
    norms = np.linalg.norm(X, axis=1)
    weights_importance = np.log(norms + 1e-10) 
    
    gd_importance = WeightedGD(
        X, Y, alpha, n_iterations, 'constant', 'orthogonal', 
        'bernoulli', weights_importance, batch_size, n_simulations
    )
    _, _, _, _, dist_importance = gd_importance.simulate_weighted_gd()

    # --- 4. Generate Plots ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(dist_uniform, label='Uniform Sampling (Standard SGD)', linewidth=2, alpha=0.8)
    plt.semilogy(dist_importance, label=r'Importance Sampling ($\propto ||X_i||$)', linewidth=2, linestyle='--')
    
    plt.xlabel('Iteration')
    plt.ylabel(r'Expected Squared Error $\mathbb{E}[||\hat{w}_k - \hat{w}||_2^2]$')
    plt.title('Convergence Speed: Uniform vs. Importance Sampling')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/speed_comparison.png')
    print("Saved speed_comparison.png")

    plt.figure(figsize=(10, 6))
    plt.semilogy(dist_importance, label='Simulated Trajectory', linewidth=2, color='#1f77b4')
    
    floor_value = np.mean(dist_importance[-50])
    plt.axhline(y=floor_value, color='red', linestyle=':', linewidth=2, label='Asymptotic Variance Floor')
    
    plt.xlabel('Iteration ($k$)')
    plt.ylabel(r'Expected Squared Error $\mathbb{E}[||\hat{w}_k - \hat{w}||_2^2]$')
    plt.title(r'Convergence to Asymptotic Variance (Constant Step-size $\alpha$)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/asymptotic_variance.png')
    print("Saved asymptotic_variance.png")

if __name__ == "__main__":
    run_comparison()