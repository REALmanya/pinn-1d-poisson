import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Device (CPU / GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PINN Neural Network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

# Forcing Function f(x)
def forcing_function(x):
    return (np.pi ** 2) * torch.sin(np.pi * x)

# Exact Solution (for validation)
def exact_solution(x):
    return torch.sin(np.pi * x)

# Collocation Points (Interior)
N_collocation = 1000
x_collocation = torch.linspace(0, 1, N_collocation).view(-1, 1).to(device)
x_collocation.requires_grad = True

# Boundary Points
x_bc = torch.tensor([[0.0], [1.0]], device=device)

# Model and Optimizer
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
epochs = 5000

for epoch in range(epochs):

    optimizer.zero_grad()

    # Network prediction
    u = model(x_collocation)

    # First derivative du/dx
    du_dx = torch.autograd.grad(
        u,
        x_collocation,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivative d²u/dx²
    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_collocation,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]

    # PDE residual
    residual = -d2u_dx2 - forcing_function(x_collocation)
    loss_pde = torch.mean(residual ** 2)

    # Boundary condition loss
    u_bc = model(x_bc)
    loss_bc = torch.mean(u_bc ** 2)

    # Total loss
    loss = loss_pde + loss_bc

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Total Loss: {loss.item():.3e}")

# Testing and Visualization
x_test = torch.linspace(0, 1, 200).view(-1, 1).to(device)
u_pred = model(x_test).detach().cpu()
u_exact = exact_solution(x_test.cpu())

# Create results directory
os.makedirs("results", exist_ok=True)

# Solution plot
plt.figure()
plt.plot(x_test.cpu(), u_exact, label="Exact Solution")
plt.plot(x_test.cpu(), u_pred, "--", label="PINN Prediction")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("1D Poisson Equation Solution")
plt.savefig("results/solution.png", dpi=300)
plt.show()

# Error plot
plt.figure()
plt.plot(x_test.cpu(), torch.abs(u_pred - u_exact))
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("Absolute Error")
plt.savefig("results/error.png", dpi=300)
plt.show()
