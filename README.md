# Physics-Informed Least Squares for a Simple Harmonic Oscillator

This project demonstrates one of the central ideas behind modern Scientific Machine Learning (SciML): physical laws can improve learning when observational data is noisy or incomplete. Instead of relying entirely on data, the model is encouraged to satisfy the governing differential equation of the system itself.

This project does **not** implement a full neural-network Physics-Informed Neural Network (PINN). Instead, it uses a simplified **physics-informed least-squares** framework. The goal is to make the core idea behind physics-informed learning transparent: a model should fit the data, but it should also obey the physics.

## The Physical System

We study the classical simple harmonic oscillator governed by

$$
\frac{d^2x}{dt^2}+x=0
$$

This equation appears in many physical systems where a restoring force drives the system back toward equilibrium.

## Analytical Solution

Assume an exponential trial solution:

$$
x(t)=e^{rt}
$$

Then

$$
\frac{d^2x}{dt^2}=r^2e^{rt}
$$

Substituting into the oscillator equation gives

$$
r^2e^{rt}+e^{rt}=0
$$

Factoring gives

$$
e^{rt}(r^2+1)=0
$$

Since

$$
e^{rt}\neq 0
$$

the characteristic equation becomes

$$
r^2+1=0
$$

Therefore,

$$
r=\pm i
$$

Using Euler's identity,

$$
e^{it}=\cos(t)+i\sin(t)
$$

the real-valued general solution becomes

$$
x(t)=A\cos(t)+B\sin(t)
$$

Applying the initial conditions

$$
x(0)=1,\qquad x'(0)=0
$$

gives

$$
A=1,\qquad B=0
$$

Therefore the exact physical solution is

$$
x(t)=\cos(t)
$$

## Noisy Observations

Sparse noisy observations are generated from the analytical solution:

```python
x_data = np.cos(t_data) + noise
```

The noise level is intentionally large enough that a purely data-driven model struggles to recover the underlying physical trajectory.

![True solution and noisy data](figures/01_true_solution_and_noisy_data.png)

## Traditional Data-Only Fitting

The flexible least-squares basis model is

$$
x(t)=A\cos(t)+B\sin(t)+Ct+D
$$

The oscillatory terms represent physical behavior, while the linear terms allow the model to drift away from the true physics.

The ordinary data-fitting loss is

$$
\mathcal{L}_{data}=\frac{1}{N}\sum_{i=1}^{N}\left(x(t_i)-x_i\right)^2
$$

Equivalently,

$$
\mathcal{L}_{data}=\int_{\Omega}\left(x(t)-x_{data}(t)\right)^2\,dt
$$

This objective encourages the model to follow noisy observations closely, even when the behavior becomes nonphysical.

## Physics-Informed Least Squares

Physics-informed learning adds a penalty for violating the governing differential equation.

For the oscillator,

$$
\frac{d^2x}{dt^2}+x=0
$$

the physics residual is

$$
r(t)=\frac{d^2x}{dt^2}+x(t)
$$

A perfectly physical solution satisfies

$$
r(t)=0
$$

throughout the domain.

The physics loss is defined as

$$
\mathcal{L}_{phys}=\frac{1}{M}\sum_{j=1}^{M}\left(\frac{d^2x}{dt^2}(t_j)+x(t_j)\right)^2
$$

Equivalently,

$$
\mathcal{L}_{phys}=\int_{\Omega}\left(\frac{d^2x}{dt^2}+x\right)^2\,dt
$$

This term penalizes solutions that violate the oscillator equation.

## Initial Condition Constraint

The initial conditions are enforced through an additional penalty term:

$$
\mathcal{L}_{IC}=\left(x(0)-1\right)^2+\left(x'(0)-0\right)^2
$$

This selects the correct trajectory from the family of physically valid oscillator solutions,

$$
x(t)=A\cos(t)+B\sin(t)
$$

Without the initial condition term, multiple physically valid oscillatory solutions would still be possible.

## Full Optimization Objective

The full physics-informed least-squares objective is

$$
\mathcal{L}=\lambda_{data}\mathcal{L}_{data}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{IC}\mathcal{L}_{IC}
$$

Here, the parameters

$$
\lambda_{data},\qquad \lambda_{phys},\qquad \lambda_{IC}
$$

control the balance between fitting observations, satisfying the physics, and enforcing the initial conditions.

![Effect of physics weight](figures/03_effect_of_physics_weight.png)

## Prediction Error Reduction

The RMSE comparison shows that adding the physics constraint significantly reduces the prediction error.

The error is reduced approximately from

$$
0.42 \rightarrow 0.13
$$

using the same noisy observations.

![RMSE comparison](figures/04_horizontal_rmse_comparison.png)

## Physics Residual Reduction

The final figure measures the physics residual norm,

$$
\left\|\frac{d^2x}{dt^2}+x\right\|
$$

As the physics weight increases, the residual decreases, showing that the learned solution increasingly satisfies the governing differential equation itself.

![Physics residual comparison](figures/05_lollipop_physics_residual.png)

## Conclusion

This project demonstrates the core intuition behind physics-informed learning: physical laws can act as powerful regularizers.

Although this project uses a simplified least-squares formulation rather than a full neural-network PINN, it captures the main idea behind PINNs and Scientific Machine Learning in a transparent and mathematically interpretable way.
