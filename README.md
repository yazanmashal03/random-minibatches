# Randomly Weighted Gradient Descent Simulation

## Description

This project simulates the iterates of the Randomly Weighted Gradient Descent (WGD) algorithm for linear regression, as described in Equation (3) of the paper "Randomly Weighted Gradient Descent in the Linear Model" by Gabriel Clara and Yazan Mash'al.

The simulation implements the following update rule:

w_hat_{k+1} = (I - alpha_k * X^T D_k^2 X) w_hat_k + alpha_k * X^T D_k^2 Y

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

1.  Configure the simulation parameters (see Configuration section below).
2.  Run the main simulation script:
    ```bash
    python simulate_wgd.py
    ```
    *(Replace `simulate_wgd.py` with the actual name of your script)*

## Input Parameters

The simulation requires the following inputs:

1.  **Data Matrix `X`**: An `n x d` NumPy array representing `n` data points with `d` features.
2.  **Target Vector `Y`**: An `n x 1` NumPy array representing the target values for the data points.
3.  **Initial Weight Vector `w_hat_1`**: A `d x 1` NumPy array for the starting point of the iteration. Often initialized to zeros or small random values.
4.  **Step Size Schedule `alpha_k`**:
    * **Constant:** A single float value (e.g., `0.01`).
    * **Decaying:** A function or rule that generates `alpha_k` based on `k` (e.g., `alpha / k`, `alpha / sqrt(k)`).
5.  **Weighting Matrix Generation `D_k`**: A function or rule to generate the sequence of diagonal matrices `D_k`. Common choices for the diagonal entries `(D_k)_{ii}`:
    * **Bernoulli (Subsampling):** `(D_k)_{ii} ~ Bernoulli(p)` independently for some `p` (often `p = b/n` for batch size `b`). The matrix `D_k` itself isn't typically constructed; instead, rows of `X` and `Y` are sampled. The formula uses `D_k^2`, so for `{0, 1}` weights, `D_k^2 = D_k`. For SGD simulating mini-batching (size `b`), one might set `b` diagonal entries of `D_k` to `sqrt(n/b)` (to keep scale) and the rest to 0, chosen uniformly. *Careful implementation is needed based on the exact SGD variant being modelled.*
    * **Gaussian:** `(D_k)_{ii} ~ N(mu, sigma^2)`.
    * **Other Distributions:** Defined according to the specific scenario being modelled.
    * Note: The equation uses `D_k^2`. Ensure your implementation squares the diagonal elements appropriately if `D_k`'s entries aren't just `0` or `1`.
6.  **Number of Iterations `K`**: An integer specifying how many iterations to run.

## Configuration

Simulation parameters (data source, initial weights, step size rule, `D_k` generation, number of iterations) should be set within the main script (`simulate_wgd.py`) or loaded from a configuration file (e.g., `config.yaml` or `config.json`).

* **[Specify where parameters are set, e.g.,:]** Parameters can be adjusted in the `config` dictionary/section at the beginning of `simulate_wgd.py`.

## Output

The simulation typically outputs:

* The sequence of iterates `w_hat_k` (potentially saved to a file, e.g., `.npy`).
* The final iterate `w_hat_K`.
* (Optional) Plots showing the convergence behavior (e.g., `||w_hat_k - w*||` if a ground truth `w*` is known, or the value of the loss function over iterations).
* (Optional) Console output logging progress or final results.

## Example

```python
# Example setup within simulate_wgd.py (Illustrative)

import numpy as np

# --- Configuration ---
N_ITERATIONS = 1000
ALPHA_RULE = lambda k: 0.01 / (k + 1) # Decaying step size
INITIAL_W = np.zeros((d, 1)) # Assuming d is defined

def generate_Dk_squared(n, p):
  # Example: Bernoulli sampling (Dk^2 = Dk for 0/1)
  # More commonly implemented by *sampling rows* directly
  # This function would return the *diagonal* of Dk^2
  diag_elements = np.random.binomial(1, p, size=n)
  return np.diag(diag_elements) # Returns the actual diagonal matrix for Dk^2

# --- Data (Load or Generate) ---
# Example: Load n, d, X, Y here
n, d = 100, 10
X = np.random.randn(n, d)
Y = X @ np.random.randn(d, 1) + np.random.randn(n, 1) * 0.1

# --- Simulation Loop ---
w_hat = INITIAL_W
iterates = [w_hat]
XT = X.T # Precompute transpose

for k in range(N_ITERATIONS):
  alpha_k = ALPHA_RULE(k)
  Dk2_diag = generate_Dk_squared_diagonal(n, p=0.1) # Get diagonal of Dk^2
  Dk2 = np.diag(Dk2_diag) # Construct diagonal matrix Dk^2

  # Calculate terms for update rule (Equation 3)
  XTDk2X = XT @ Dk2 @ X
  XTDk2Y = XT @ Dk2 @ Y

  # Update w_hat
  w_hat = (np.identity(d) - alpha_k * XTDk2X) @ w_hat + alpha_k * XTDk2Y
  iterates.append(w_hat)

# --- Output/Save Results ---
final_w = iterates[-1]
print(f"Final weights: {final_w.flatten()}")
# Optionally save iterates, plot convergence etc.