import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# REAL PYTORCH PINN DEMO: SIMPLE HARMONIC OSCILLATOR
#
# Physics equation:
#       x'' + x = 0
#
# Analytical solution:
#       x(t) = cos(t)
#
# PINN objective:
#       L = lambda_NN L_NN + lambda_phys L_phys + lambda_IC L_IC
#
# This script generates:
#   01_true_solution_and_noisy_data.png
#   02_main_comparison.png
#   03_effect_of_physics_weight.png
#   04_horizontal_rmse_comparison.png
#   05_lollipop_physics_residual.png
# ============================================================

np.random.seed(10)
torch.manual_seed(10)

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Generate analytical solution and noisy observations
# ------------------------------------------------------------

t_min = 0.0
t_max = 4 * np.pi

t_true = np.linspace(t_min, t_max, 400).reshape(-1, 1)
x_true = np.cos(t_true)

n_data = 28
noise_level = 0.45

t_data = np.linspace(t_min, t_max, n_data).reshape(-1, 1)
x_data = np.cos(t_data) + noise_level * np.random.randn(n_data, 1)

# Torch tensors
t_data_t = torch.tensor(t_data, dtype=torch.float32)
x_data_t = torch.tensor(x_data, dtype=torch.float32)

t_true_t = torch.tensor(t_true, dtype=torch.float32)

# Physics collocation points
n_phys = 160
t_phys_base = torch.linspace(t_min, t_max, n_phys).reshape(-1, 1)

# ------------------------------------------------------------
# 2. Define neural network
# ------------------------------------------------------------

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

def second_derivative(y, t):
    """
    Compute d²y/dt² using PyTorch automatic differentiation.
    """

    dy_dt = torch.autograd.grad(
        y,
        t,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    d2y_dt2 = torch.autograd.grad(
        dy_dt,
        t,
        grad_outputs=torch.ones_like(dy_dt),
        create_graph=True
    )[0]

    return d2y_dt2

mse = nn.MSELoss()

# ------------------------------------------------------------
# 3. Train data-only neural network
# ------------------------------------------------------------

data_nn = NeuralNet()
optimizer = torch.optim.Adam(data_nn.parameters(), lr=1e-3)

data_epochs = 5000

for epoch in range(data_epochs):

    optimizer.zero_grad()

    x_pred = data_nn(t_data_t)

    loss = mse(x_pred, x_data_t)

    loss.backward()

    optimizer.step()

with torch.no_grad():
    x_data_nn = data_nn(t_true_t).numpy()

rmse_data_nn = np.sqrt(np.mean((x_data_nn - x_true) ** 2))

# ------------------------------------------------------------
# 4. Function to train PINN for a chosen physics weight
# ------------------------------------------------------------

def train_pinn(lambda_phys, lambda_NN=1.0, lambda_IC=10.0, epochs=8000):

    model = NeuralNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    x0 = torch.tensor([[1.0]], dtype=torch.float32)
    v0 = torch.tensor([[0.0]], dtype=torch.float32)

    loss_history = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        # ----------------------------
        # NN/data loss
        # ----------------------------
        x_pred_data = model(t_data_t)

        loss_NN = mse(x_pred_data, x_data_t)

        # ----------------------------
        # Physics loss
        # ----------------------------
        t_phys = t_phys_base.clone().detach().requires_grad_(True)

        x_pred_phys = model(t_phys)

        x_tt = second_derivative(x_pred_phys, t_phys)

        residual = x_tt + x_pred_phys

        loss_phys = mse(residual, torch.zeros_like(residual))

        # ----------------------------
        # Initial condition loss
        # ----------------------------
        x_pred_0 = model(t0)

        dx_dt_0 = torch.autograd.grad(
            x_pred_0,
            t0,
            grad_outputs=torch.ones_like(x_pred_0),
            create_graph=True
        )[0]

        loss_IC = mse(x_pred_0, x0) + mse(dx_dt_0, v0)

        # ----------------------------
        # Total loss
        # ----------------------------
        loss = (
            lambda_NN * loss_NN
            + lambda_phys * loss_phys
            + lambda_IC * loss_IC
        )

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            loss_history.append(loss.item())

    # Predictions
    with torch.no_grad():
        x_pred = model(t_true_t).numpy()

    rmse = np.sqrt(np.mean((x_pred - x_true) ** 2))

    # Physics residual on evaluation grid
    t_eval = torch.tensor(t_true, dtype=torch.float32, requires_grad=True)

    x_eval = model(t_eval)

    x_eval_tt = second_derivative(x_eval, t_eval)

    residual_eval = x_eval_tt + x_eval

    residual_rmse = torch.sqrt(torch.mean(residual_eval ** 2)).item()

    return model, x_pred, rmse, residual_rmse, loss_history

# ------------------------------------------------------------
# 5. Train PINNs for multiple physics weights
# ------------------------------------------------------------

lambda_phys_values = [0.01, 0.1, 1.0, 10.0, 100.0]

pinn_predictions = {}
pinn_rmse = {}
pinn_residual = {}

for lambda_phys in lambda_phys_values:

    print(f"Training PINN with lambda_phys={lambda_phys}...")

    model, pred, rmse, residual_rmse, loss_history = train_pinn(
        lambda_phys=lambda_phys,
        lambda_NN=1.0,
        lambda_IC=10.0,
        epochs=8000
    )

    pinn_predictions[lambda_phys] = pred
    pinn_rmse[lambda_phys] = rmse
    pinn_residual[lambda_phys] = residual_rmse

# ------------------------------------------------------------
# 6. Graph 1: True solution and noisy data
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    t_true,
    x_true,
    linewidth=3,
    label="Analytical solution: cos(t)"
)

ax.scatter(
    t_data,
    x_data,
    s=60,
    alpha=0.75,
    label="Noisy observations"
)

ax.set_xlabel("Time, t")
ax.set_ylabel("Displacement, x(t)")
ax.set_title("Analytical Solution and Noisy Observations")
ax.grid(True, alpha=0.3)
ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(output_dir / "01_true_solution_and_noisy_data.png", dpi=240, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 7. Graph 2: Main comparison
# ------------------------------------------------------------

best_lambda = 100.0

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    t_data,
    x_data,
    s=55,
    alpha=0.75,
    label="Noisy observations"
)

ax.plot(
    t_true,
    x_true,
    linewidth=3,
    label="Analytical solution"
)

ax.plot(
    t_true,
    x_data_nn,
    "--",
    linewidth=2,
    label=f"Data-only NN, RMSE={rmse_data_nn:.3f}"
)

ax.plot(
    t_true,
    pinn_predictions[best_lambda],
    linewidth=3,
    label=f"PINN, RMSE={pinn_rmse[best_lambda]:.3f}"
)

ax.set_xlabel("Time, t")
ax.set_ylabel("Displacement, x(t)")
ax.set_title("Data-Only Neural Network vs Physics-Informed Neural Network")
ax.grid(True, alpha=0.3)
ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(output_dir / "02_main_comparison.png", dpi=240, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 8. Graph 3: Effect of physics weight
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    t_data,
    x_data,
    s=45,
    alpha=0.55,
    label="Noisy observations"
)

ax.plot(
    t_true,
    x_true,
    linewidth=3,
    label="Analytical solution"
)

for lambda_phys in [0.01, 0.1, 1.0, 100.0]:

    ax.plot(
        t_true,
        pinn_predictions[lambda_phys],
        linewidth=2,
        label=f"lambda_phys={lambda_phys}"
    )

ax.set_xlabel("Time, t")
ax.set_ylabel("Displacement, x(t)")
ax.set_title("Effect of Physics Weight on the Learned Solution")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(output_dir / "03_effect_of_physics_weight.png", dpi=240, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 9. Graph 4: Horizontal RMSE comparison
# ------------------------------------------------------------

labels = ["Data-only NN"] + [
    f"PINN lambda_phys={value}" for value in lambda_phys_values
]

rmse_all = np.array([rmse_data_nn] + [
    pinn_rmse[value] for value in lambda_phys_values
])

labels_rev = labels[::-1]
rmse_rev = rmse_all[::-1]

y_pos = np.arange(len(labels_rev))

fig, ax = plt.subplots(figsize=(10, 5.8))

ax.barh(
    y_pos,
    rmse_rev,
    height=0.55,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.85
)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_rev)
ax.set_xlabel("RMSE against analytical solution")
ax.set_title("Prediction Error: Data-Only NN vs PINN Models", pad=14)

for y, value in zip(y_pos, rmse_rev):

    ax.text(
        value + 0.008,
        y,
        f"{value:.3f}",
        va="center",
        fontsize=10
    )

ax.set_xlim(0, max(rmse_all) * 1.22)
ax.grid(axis="x", linestyle="--", alpha=0.25)

ax.text(
    0.98,
    0.08,
    "Lower is better",
    transform=ax.transAxes,
    ha="right",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.12)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.28, right=0.96, top=0.88, bottom=0.14)

plt.savefig(output_dir / "04_horizontal_rmse_comparison.png", dpi=240, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 10. Graph 5: Physics residual lollipop chart
# ------------------------------------------------------------

residual_values = np.array([
    pinn_residual[value] for value in lambda_phys_values
])

residual_labels = [
    f"lambda_phys={value}" for value in lambda_phys_values
]

labels_rev = residual_labels[::-1]
residual_rev = residual_values[::-1]

y_pos = np.arange(len(labels_rev))

fig, ax = plt.subplots(figsize=(10, 5.8))

ax.hlines(
    y=y_pos,
    xmin=1e-5,
    xmax=residual_rev,
    linewidth=3,
    alpha=0.55
)

ax.scatter(
    residual_rev,
    y_pos,
    s=90,
    edgecolor="black",
    linewidth=0.8,
    zorder=3
)

ax.set_xscale("log")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_rev)
ax.set_xlabel("Physics residual RMSE: ||x'' + x||")
ax.set_title("Physics Consistency Improves as Constraint Weight Increases", pad=14)

for y, value in zip(y_pos, residual_rev):

    ax.text(
        value * 1.18,
        y,
        f"{value:.4f}",
        va="center",
        fontsize=10
    )

ax.grid(axis="x", which="both", linestyle="--", alpha=0.25)

ax.text(
    0.98,
    0.08,
    "Lower residual = more physics-consistent",
    transform=ax.transAxes,
    ha="right",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.12)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.28, right=0.96, top=0.88, bottom=0.14)

plt.savefig(output_dir / "05_lollipop_physics_residual.png", dpi=240, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 11. Print summary
# ------------------------------------------------------------

print("\n===== SUMMARY =====\n")

print(f"Noise level: {noise_level}")
print(f"Data-only NN RMSE: {rmse_data_nn:.4f}")

for value in lambda_phys_values:
    print(
        f"lambda_phys={value:>6}: "
        f"PINN RMSE={pinn_rmse[value]:.4f}, "
        f"Physics residual RMSE={pinn_residual[value]:.6f}"
    )

print(f"\nFigures saved in: {output_dir.resolve()}")
