# Randomly Weighted Gradient Descent Simulation

This project implements a simulation of randomly weighted gradient descent as described in equation (3) of the paper "Randomly Weighted Gradient Descent in the Linear Model". The implementation demonstrates how different batch sizes affect the convergence of gradient descent in a linear regression setting.

## Equation Description

The simulation implements the following gradient descent update rule:

```
w_{k+1} = w_k - α_k * X^T * D_k * D_k * (X * w_k - Y)
```

where:
- w_k is the weight vector at iteration k
- α_k is the step size
- X is the data matrix
- Y is the target vector
- D_k is a random diagonal matrix that weights the samples

## Requirements

- Python 3.6+
- numpy (>=1.21.0)
- matplotlib (>=3.4.0)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation with:
```bash
python simulate_gd.py
```

This will:
1. Generate synthetic data for a linear regression problem
2. Run gradient descent with different initializations:
   - Zero initialization
   - Random initialization
3. Plot the norm of the weights over iterations
4. Save the plot as 'overparametrized_gd.png'
5. Print diagnostics about the data matrix

## Parameters

The simulation can be customized by modifying the following parameters in `simulate_gd.py`:

- `n_samples`: Number of data points (default: 50)
- `n_features`: Number of features (default: 100)
- `n_iterations`: Number of gradient descent iterations (default: 100)
- `alpha`: Step size (default: 0.01)
- `initialization`: Weight initialization method ('zero' or 'random')

## Overparametrized Setting

In the overparametrized setting (where number of features > number of samples), gradient descent can exhibit interesting behavior:

1. **Blow-up Phenomenon**: The weights can grow unboundedly large, especially with random initialization. This happens because:
   - The matrix X^T X is singular (not invertible)
   - There are infinitely many solutions that perfectly fit the training data
   - The gradient descent path can diverge if not properly constrained

2. **Initialization Dependence**: The behavior strongly depends on initialization:
   - Zero initialization tends to find the minimum-norm solution
   - Random initialization can lead to divergent behavior

3. **Diagnostics**: The script prints important diagnostics:
   - Condition number of X^T X (indicates numerical stability)
   - Rank of X (indicates linear dependencies)
   - Number of samples and features

## Output

The script generates a plot showing the norm of the weights over iterations for different initializations. This helps visualize how the weights evolve in the overparametrized setting and demonstrates the potential for blow-up.
