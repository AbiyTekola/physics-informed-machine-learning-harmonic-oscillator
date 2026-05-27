# Physics-Informed Least Squares for a Simple Harmonic Oscillator

This project demonstrates one of the central ideas behind modern Scientific Machine Learning (SciML): physical laws can improve learning when observational data is noisy or incomplete. Instead of relying entirely on data, the model is encouraged to satisfy the governing differential equation of the system itself. Even in a very simple oscillator problem, incorporating physics significantly reduces overfitting and produces solutions that remain much closer to the true underlying behavior.

This project does **not** implement a full neural-network Physics-Informed Neural Network (PINN). Instead, it uses a simplified **physics-informed least-squares** framework. The purpose is pedagogical clarity: to isolate and visualize how physics constraints suppress nonphysical behavior and improve stability.

---

## The Physical System

We study the classical simple harmonic oscillator governed by

$$
\frac{d^2x}{dt^2}+x=0
$$

This equation appears whenever a restoring force pulls a system back toward equilibrium. Examples include mass-spring systems, pendulums at small angles, LC electrical circuits, molecular vibrations, and many wave phenomena.

The key physical idea is simple but profound: acceleration always points opposite to displacement. When the displacement becomes positive, the acceleration becomes negative, driving the system back toward equilibrium. Once the system crosses equilibrium, the sign reverses again. This continual exchange between displacement and restoring acceleration produces oscillatory motion.

Mathematically, the second derivative controls curvature. The equation therefore constrains not only the position of the trajectory but also how the trajectory bends throughout time. This makes the oscillator an ideal test problem for physics-informed learning.

---

## Analytical Solution

To solve the differential equation analytically, assume an exponential trial solution:

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

Factoring out the exponential term gives

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

This analytical solution serves as the ground-truth reference throughout the experiment.

---

## Noisy Observations

Real scientific measurements are rarely perfect. Experimental data often includes sensor noise, calibration uncertainty, environmental disturbances, and incomplete sampling. To mimic this situation, sparse noisy observations are generated directly from the analytical solution:

```python
x_data = np.cos(t_data) + noise
```

The noise level is intentionally large enough that a purely data-driven model struggles to recover the underlying physical trajectory.

<p align="center">
  <img src="figures/01_true_solution_and_noisy_data.png" width="520">
</p>

---

## Traditional Data-Only Fitting

A purely data-driven model attempts only to minimize disagreement with the observations. In this project, the flexible basis model is

$$
x(t)=A\cos(t)+B\sin(t)+Ct+D
$$

The oscillatory terms naturally represent physically plausible behavior, while the additional linear terms allow the model to drift away from the true governing dynamics.

The ordinary least-squares loss is

$$
\mathcal{L}_{data}=\frac{1}{N}\sum_{i=1}^{N}\left(x(t_i)-x_i\right)^2
$$

Equivalently, in continuous form,

$$
\mathcal{L}_{data}=\int_{\Omega}\left(x(t)-x_{data}(t)\right)^2\,dt
$$

This objective encourages the model to match noisy observations as closely as possible. However, by itself, it contains no knowledge of the governing differential equation. As a result, the fitted curve can follow noise and develop nonphysical behavior.

---

## Physics-Informed Least Squares

Physics-informed learning adds a penalty for violating the governing differential equation itself.

For the oscillator,

$$
\frac{d^2x}{dt^2}+x=0
$$

the physics residual becomes

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

This term penalizes solutions that violate the oscillator equation. The model is therefore encouraged not only to fit the data but also to remain physically admissible.

---

## Initial Condition Constraint

The initial conditions are enforced through an additional penalty term:

$$
\mathcal{L}_{IC}=\left(x(0)-1\right)^2+\left(x'(0)-0\right)^2
$$

This selects the correct trajectory from the family of physically valid oscillator solutions,

$$
x(t)=A\cos(t)+B\sin(t)
$$

Without the initial-condition term, multiple physically valid oscillatory solutions would still remain possible.

---

## Full Optimization Objective

The full physics-informed least-squares objective becomes

$$
\mathcal{L}=\lambda_{data}\mathcal{L}_{data}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{IC}\mathcal{L}_{IC}
$$

The parameters

$$
\lambda_{data},\qquad \lambda_{phys},\qquad \lambda_{IC}
$$

control the balance between fitting observations, satisfying the governing physics, and enforcing the initial conditions.

When the data term dominates, the model behaves more like a traditional least-squares fit and aggressively follows noisy measurements. As the physics weight increases, the solution increasingly sacrifices local noisy fitting in favor of globally consistent physical behavior.

---

## Effect of the Physics Constraint

The next figure shows how the learned trajectory changes as the physics weight increases. With a weak physics penalty, the model still follows noisy fluctuations. As the physics constraint strengthens, the learned trajectory becomes smoother and more physically realistic.

<p align="center">
  <img src="figures/03_effect_of_physics_weight.png" width="520">
</p>

The transition is important conceptually. The model moves from trying to reproduce every noisy fluctuation toward identifying the most physically plausible explanation of the observations.

---

## Prediction Error Reduction

The RMSE comparison demonstrates a substantial reduction in prediction error once the physics constraint is introduced. The purely data-driven model produces the largest error because it aggressively follows noise, while the physics-informed solution stays significantly closer to the true oscillator.

In this experiment, the prediction error is reduced approximately from

$$
0.42 \rightarrow 0.13
$$

despite using the exact same noisy observations.

<p align="center">
  <img src="figures/04_horizontal_rmse_comparison.png" width="480">
</p>

This illustrates how governing equations can function as strong regularizers in scientific learning problems.

---

## Physics Residual Reduction

The final figure measures the norm of the physics residual,

$$
\left\|\frac{d^2x}{dt^2}+x\right\|
$$

As the physics weight increases, the residual decreases significantly, showing that the learned solution increasingly satisfies the differential equation itself.

<p align="center">
  <img src="figures/05_lollipop_physics_residual.png" width="480">
</p>

This is important because the model is not merely producing a visually smooth curve. It is becoming more consistent with the underlying governing physics.

---

## Interpretation

The central lesson is that physics can act as a powerful inductive bias. A purely data-driven model searches for any curve that matches noisy observations, whereas a physics-informed model searches for a curve that both matches the observations and satisfies the governing differential equation.

This project uses a simplified least-squares formulation, but the same conceptual structure appears in full PINNs. In neural-network PINNs, the model output is produced by a neural network and derivatives are computed using automatic differentiation. Here, the simplified least-squares version makes the mechanism easier to inspect mathematically.

---

## Conclusion

This small experiment captures one of the foundational ideas behind Scientific Machine Learning: physical laws can dramatically improve learning in noisy environments.

Traditional machine-learning models search over a vast space of arbitrary functions. Physics-informed methods instead restrict the search to functions that are simultaneously data-consistent, mathematically smooth, and physically admissible.

Although this project uses a simplified least-squares formulation rather than a full neural-network PINN, it demonstrates the core intuition behind physics-informed learning in a transparent and mathematically interpretable way.
