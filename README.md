# Randomly Weighted Gradient Descent Simulation

## Description

This project simulates the iterates of the Randomly Weighted Gradient Descent (WGD) algorithm for linear regression, as described in Equation (3) of the paper "Randomly Weighted Gradient Descent in the Linear Model" by Gabriel Clara and Yazan Mash'al.

The simulation implements the following update rule:

ŵₖ₊₁ = (I - αₖ Xᵀ Dₖ² X) ŵₖ + αₖ Xᵀ Dₖ² Y

where:
* `w_hat_k` is the weight vector estimate at iteration `k`.
* `alpha_k` is the step size at iteration `k`.
* `X` is the `n x d` data matrix (features).
* `Y` is the `n x 1` target vector.
* `D_k` is a random `n x n` diagonal weighting matrix at iteration `k`.
* `I` is the `d x d` identity matrix.

## Requirements

* Python 3.x
* NumPy (`pip install numpy`)
* (Optional) Matplotlib (`pip install matplotlib`) for visualization
* (Optional) SciPy (`pip install scipy`) for more complex data generation or analysis

## Usage
