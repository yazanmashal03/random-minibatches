# Randomly Weighted Gradient Descent Simulation

This repository implements and empirically validates the theoretical results for randomly weighted gradient descent from the paper. The implementation focuses on verifying the convergence bounds for both first and second moments of the iterates.

## Mathematical Background

### Problem Setting

We consider the weighted least squares problem:

$$
\min_{w \in \mathbb{R}^d} \frac{1}{2} \|D(Xw - Y)\|^2
$$

where:
- $X \in \mathbb{R}^{n \times d}$ is the data matrix
- $Y \in \mathbb{R}^n$ is the target vector
- $D \in \mathbb{R}^{n \times n}$ is a random diagonal matrix with i.i.d. entries

### Algorithm

The randomly weighted gradient descent algorithm updates the weights according to:

$$
w_{k+1} = (I-\alpha_k \cdot X^t D_k^2X) \hat{w_k} + \alpha_k \cdot X^T D_k^2 Y
$$

where $\alpha_k$ is the step size at iteration k and $D_k$ is independently sampled at each iteration.

### Theoretical Results

#### First Moment Convergence (Lemma 3.2)
The expected iterates converge to the weighted least squares solution $\hat{w}$ at a linear rate:

$$
\|\mathbb{E}[w_k - \hat{w}]\| \leq \exp(-\sigma_{\min}^+(\mathscr{X})\sum_{\ell=1}^k \alpha_\ell) \|w_1 - \hat{w}\|
$$

#### Second Moment Convergence (Lemma 3.3)
The second moment convergence is characterized by the operator $S_\alpha$ defined as:

$$
S_\alpha(A) = (I - \alpha \hat{X})A(I - \alpha \hat{X}) + \alpha^2 X^T(\Sigma_D \circ (XAX^T + (Y-X\hat{w})(Y-X\hat{w})^T))X
$$

where $\hat{X} = M_2^{1/2}X$ and $\Sigma_D$ is the covariance matrix of the diagonal entries of $D$.

The difference between consecutive second moments satisfies:

$$
\begin{align*}
\|\mathbb{E}[(w_{k+1} - \hat{w})(w_{k+1} - \hat{w})^T] - S_{\alpha_k}(\mathbb{E}[(w_k - \hat{w})(w_k - \hat{w})^T])\| \\
\leq 2\alpha_k^2 \|X\|^3 \|\Sigma_D\| \exp(-\sigma_{\min}^+(X)\sum_{\ell=1}^{k-1} \alpha_\ell) \|w_1 - \hat{w}\| \|Y - X\hat{w}\|
\end{align*}
$$

## Implementation Details

### Weight Distributions
The code supports three types of random weight distributions:
1. Uniform: $D_{ii} \sim U(0,1)$
2. Bernoulli: $D_{ii} \sim \text{Bernoulli}(p)$
3. Binary: Randomly select half of diagonal entries to be 1, others 0

### Step Size Schemes
Two step size schemes are implemented:
1. Constant: $\alpha_k = \alpha$
2. Diminishing: $\alpha_k = \frac{c}{k}$ where $c$ is scaled by $\|X\|$

### Initialization
The initial weights $w_1$ are chosen to satisfy Assumption 3.1(b):
- Orthogonal to $\text{ker}(X)$
- Random initialization scaled to unit norm

## Usage

```python
from simulate_gd import simulate_weighted_gd

# Run simulation
first_moment, first_bound, second_moment, second_bound = simulate_weighted_gd(
    X, Y,
    n_iterations=1000,
    step_type='constant',  # or 'diminishing'
    initialization='orthogonal',
    weight_distribution='uniform',  # or 'bernoulli' or 'binary'
    p=0.2,  # probability for bernoulli
    n_simulations=50
)
```

## Results

The implementation produces plots comparing:
1. First moment convergence vs theoretical bound
2. Second moment convergence vs theoretical bound
for both constant and diminishing step sizes.

## Requirements
- numpy
- matplotlib
- scipy
