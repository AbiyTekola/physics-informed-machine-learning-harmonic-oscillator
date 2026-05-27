# Physics-Informed Least Squares for a Simple Harmonic Oscillator

This project demonstrates one of the central ideas behind modern Scientific Machine Learning (SciML): physical laws can dramatically improve learning when observational data is noisy or incomplete. Instead of relying entirely on data, the model is encouraged to satisfy the governing differential equation of the system itself. Even in a very simple oscillator problem, incorporating physics significantly reduces overfitting and produces solutions that remain much closer to the true underlying behavior.

Unlike a full Physics-Informed Neural Network (PINN), this project uses a simplified physics-informed least-squares framework. The goal is not to build a large deep-learning system, but rather to isolate the core idea behind physics-informed learning in the clearest possible way. By using a compact analytical basis model, the role of the physics constraint becomes transparent and easy to visualize.

## The Physical System

We study the classical simple harmonic oscillator governed by:

$$
\frac{d^2x}{dt^2}+x=0
$$

This equation appears throughout physics whenever a restoring force drives a system back toward equilibrium. Examples include mass-spring systems, small-angle pendulums, electrical LC circuits, molecular vibrations, and many wave phenomena.

The negative relationship between displacement and acceleration produces oscillatory motion. Whenever the displacement becomes positive, the acceleration points in the opposite direction, continuously pulling the system back toward equilibrium.

## Analytical Solution

To solve the equation analytically, we assume an exponential trial solution:

$$
x(t)=e^{rt}
$$

Differentiating twice gives:

$$
\frac{d^2x}{dt^2}=r^2e^{rt}
$$

Substituting into the differential equation yields:

$$
r^2e^{rt}+e^{rt}=0
$$

Factoring out the exponential term:

$$
e^{rt}(r^2+1)=0
$$

Since the exponential is never zero, the characteristic equation becomes:

$$
r^2+1=0
$$

which gives the imaginary roots:

$$
r=\pm i
$$

Using Euler's identity,

$$
e^{it}=\cos(t)+i\sin(t)
$$

the real-valued general solution becomes:

$$
x(t)=A\cos(t)+B\sin(t)
$$

Applying the initial conditions

$$
x(0)=1,\qquad x'(0)=0
$$

gives:

$$
A=1,\qquad B=0
$$

so the exact physical solution is:

$$
x(t)=\cos(t)
$$

This analytical solution serves as the ground-truth reference throughout the experiment.

## Noisy Observations

Real scientific measurements are rarely perfect. Experimental data usually contains sensor noise, calibration uncertainty, environmental disturbances, and incomplete sampling. To mimic this situation, sparse noisy observations were generated directly from the analytical solution:

```python
x_data = np.cos(t_data) + noise
```

The noise level was intentionally made large so that a purely data-driven fit would struggle to recover the underlying physical behavior. This creates a meaningful test for physics-informed learning.

The first figure shows the exact analytical oscillator together with the noisy observations used for fitting.

![True solution and noisy data](figures/01_true_solution_and_noisy_data.png)

## Traditional Data-Only Fitting

A purely data-driven model attempts only to minimize disagreement with the observations. In this project, the following flexible basis model was used:

$$
x(t)=A\cos(t)+B\sin(t)+Ct+D
$$

The oscillatory terms naturally represent physical oscillator behavior, while the linear terms

$$
Ct+D
$$

allow the model to drift away from the correct physics.

The ordinary least-squares loss becomes:

$$
\mathcal{L}_{data}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(x(t_i)-x_i\right)^2
$$

or, equivalently in continuous form,

$$
\mathcal{L}_{data}
=
\int_{\Omega}
\left(x(t)-x_{data}(t)\right)^2
\,dt
$$

This objective encourages the model to match the noisy observations as closely as possible, but it contains no knowledge of the governing physics. As a result, the solution often follows noise aggressively and develops nonphysical behavior.

## Physics-Informed Least Squares

The central idea behind physics-informed learning is to penalize violations of the governing differential equation directly.

For the oscillator equation,

$$
\frac{d^2x}{dt^2}+x=0
$$

the physics residual becomes:

$$
r(t)=\frac{d^2x}{dt^2}+x(t)
$$

A perfectly physical solution should satisfy

$$
r(t)=0
$$

everywhere throughout the domain.

The physics loss is therefore defined as:

$$
\mathcal{L}_{phys}
=
\frac{1}{M}
\sum_{j=1}^{M}
\left(
\frac{d^2x}{dt^2}(t_j)+x(t_j)
\right)^2
$$

or, equivalently,

$$
\mathcal{L}_{phys}
=
\int_{\Omega}
\left(
\frac{d^2x}{dt^2}+x
\right)^2
\,dt
$$

This term penalizes any solution that violates the differential equation. In effect, the model is no longer allowed to fit arbitrary curves; it is encouraged to remain physically admissible.

## Initial Conditions

The initial conditions are enforced through an additional penalty term:

$$
\mathcal{L}_{IC}
=
\left(x(0)-1\right)^2
+
\left(x'(0)-0\right)^2
$$

This selects the correct trajectory from the family of oscillator solutions:

$$
x(t)=A\cos(t)+B\sin(t)
$$

Without this term, many physically valid oscillatory solutions would still remain possible.

## Full Optimization Objective

The complete physics-informed optimization problem becomes:

$$
\mathcal{L}
=
\lambda_{data}\mathcal{L}_{data}
+
\lambda_{phys}\mathcal{L}_{phys}
+
\lambda_{IC}\mathcal{L}_{IC}
$$

The three weights control the balance between fitting observations, satisfying the governing differential equation, and enforcing the initial conditions.

This balance is the core idea behind both PINNs and many other SciML methods. When the data term dominates, the model behaves more like a traditional least-squares fit and aggressively follows noisy measurements. As the physics weight increases, the solution increasingly sacrifices noisy local fitting in favor of globally consistent physical behavior.

## Effect of the Physics Constraint

The next figure shows how the learned trajectory changes as the physics weight increases. With small physics weights, the solution still exhibits noticeable overfitting and follows noise too closely. As the physics penalty becomes stronger, the trajectory becomes smoother and more physically realistic.

The model gradually transitions from attempting to "fit the noisy observations" toward finding the "best physically admissible explanation" of those observations.

![Effect of physics weight](figures/03_effect_of_physics_weight.png)

## Prediction Error Reduction

The RMSE comparison demonstrates a substantial reduction in prediction error once physics constraints are introduced.

The purely data-driven model produces the largest error because it aggressively follows noise. As the physics constraint strengthens, the solution becomes more stable and significantly closer to the true oscillator.

In this experiment, the strongest physics-informed solution reduced the prediction error approximately from:

$$
0.42 \rightarrow 0.13
$$

despite using exactly the same noisy observations.

![RMSE comparison](figures/04_horizontal_rmse_comparison.png)

This illustrates how governing equations can function as powerful regularizers in scientific learning problems.

## Physics Residual Reduction

The final figure shows the norm of the physics residual:

$$
\left\|
\frac{d^2x}{dt^2}+x
\right\|
$$

As the physics weight increases, the residual decreases by several orders of magnitude. This is important because the model is not merely producing visually smooth curves; it is increasingly satisfying the governing differential equation itself.

![Physics residual comparison](figures/05_lollipop_physics_residual.png)

The reduction of the residual provides direct evidence that the learned solution is becoming more physically consistent.

## Conclusion

This small experiment captures one of the foundational ideas behind Scientific Machine Learning: physical laws provide a powerful inductive bias.

Traditional machine-learning models search over a vast space of arbitrary functions. Physics-informed methods instead restrict the search to functions that are simultaneously data-consistent, mathematically smooth, and physically admissible.

Even in this elementary oscillator problem, the improvement is substantial. In realistic scientific applications involving partial differential equations, sparse observations, inverse problems, fluid dynamics, climate simulations, digital twins, and computational physics, the same principle becomes even more powerful.

Although this project uses a simplified least-squares framework rather than a full neural-network PINN, it demonstrates the core intuition behind physics-informed learning in a transparent and mathematically interpretable way.
