# Physics-Informed Least Squares for a Simple Harmonic Oscillator

This project demonstrates one of the central ideas behind modern Scientific Machine Learning (SciML): physical laws can dramatically improve learning when observational data is noisy or incomplete. Instead of relying entirely on data, the model is encouraged to satisfy the governing differential equation of the system itself.

Unlike a full Physics-Informed Neural Network (PINN), this project uses a simplified physics-informed least-squares framework. The goal is pedagogical clarity: to isolate and visualize how physics constraints suppress nonphysical behavior and reduce overfitting.

## The Physical System

We study the classical simple harmonic oscillator governed by:

`d²x/dt² + x = 0`

This equation appears throughout physics whenever a restoring force drives a system back toward equilibrium.

## Analytical Solution

We assume an exponential trial solution:

`x(t) = e^(rt)`

Differentiating twice gives:

`d²x/dt² = r² e^(rt)`

Substituting into the differential equation:

`r² e^(rt) + e^(rt) = 0`

Factoring:

`e^(rt)(r² + 1) = 0`

Since the exponential is never zero:

`r² + 1 = 0`

which gives:

`r = ± i`

Using Euler’s identity:

`e^(it) = cos(t) + i sin(t)`

the general real-valued solution becomes:

`x(t) = A cos(t) + B sin(t)`

Applying the initial conditions:

`x(0) = 1`
`x'(0) = 0`

gives:

`A = 1`
`B = 0`

Therefore:

`x(t) = cos(t)`

## Noisy Observations

Sparse noisy observations were generated directly from the analytical solution:

```python
x_data = np.cos(t_data) + noise
```

The noise level was intentionally chosen to make the fitting problem challenging.

![True solution and noisy data](figures/01_true_solution_and_noisy_data.png)

## Traditional Data-Only Fitting

The flexible least-squares basis model used in this project is:

`x(t) = A cos(t) + B sin(t) + Ct + D`

The oscillatory terms represent physically valid behavior, while the linear terms allow the solution to drift away from the governing physics.

The ordinary least-squares loss becomes:

`L_data = (1/N) Σ (x(t_i) - x_i)^2`

This objective encourages the model to follow noisy observations closely, even when that behavior becomes nonphysical.

## Physics-Informed Least Squares

The central idea behind physics-informed learning is to penalize violations of the governing differential equation directly.

The physics residual is defined as:

`r(t) = d²x/dt² + x(t)`

A perfectly physical solution should satisfy:

`r(t) = 0`

throughout the domain.

The physics loss becomes:

`L_phys = (1/M) Σ (d²x/dt²(t_j) + x(t_j))²`

This term penalizes solutions that violate the governing equation.

## Initial Conditions

The initial-condition penalty term is:

`L_IC = (x(0)-1)² + (x'(0)-0)²`

This selects the correct physical trajectory from the oscillator family.

## Full Optimization Objective

The complete objective function becomes:

`L = λ_data L_data + λ_phys L_phys + λ_IC L_IC`

The three weights control the balance between:
- fitting observations,
- satisfying the differential equation,
- enforcing the initial conditions.

As the physics weight increases, the learned solution becomes smoother and more physically admissible.

![Effect of physics weight](figures/03_effect_of_physics_weight.png)

## Prediction Error Reduction

The RMSE comparison demonstrates substantial reduction in prediction error once physics constraints are introduced.

The strongest physics-informed solution reduced the prediction error approximately from:

`0.42 → 0.13`

despite using exactly the same noisy observations.

![RMSE comparison](figures/04_horizontal_rmse_comparison.png)

## Physics Residual Reduction

The norm of the physics residual is:

`|| d²x/dt² + x ||`

As the physics weight increases, the residual decreases significantly, indicating stronger agreement with the governing differential equation.

![Physics residual comparison](figures/05_lollipop_physics_residual.png)

## Conclusion

This small experiment captures one of the foundational ideas behind Scientific Machine Learning: physical laws provide a powerful inductive bias.

Traditional machine-learning models search over a vast space of arbitrary functions. Physics-informed methods instead restrict the search to functions that are simultaneously:
- data-consistent,
- mathematically smooth,
- physically admissible.

Although this project uses a simplified least-squares framework rather than a full neural-network PINN, it demonstrates the core intuition behind physics-informed learning in a transparent and mathematically interpretable way.
