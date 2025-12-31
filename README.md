# Physics-Informed Neural Network for 1D Poisson Equation

This repository reproduces a foundational Physics-Informed Neural Network (PINN)
example from Raissi et al. (2019).

## Problem Statement

We solve the one-dimensional Poisson equation:

- d²u/dx² = π² sin(πx),  x ∈ (0, 1)

with boundary conditions:

u(0) = 0,  u(1) = 0

The analytical solution is:

u(x) = sin(πx)

## Methodology

A fully-connected neural network approximates the solution u(x).
The network is trained by minimizing a composite loss consisting of:

- PDE residual loss (physics enforcement)
- Boundary condition loss

Automatic differentiation is used to compute spatial derivatives exactly.
No labeled solution data are used during training.

## Results

The PINN accurately reproduces the analytical solution.

![Solution](results/solution.png)

## Purpose

This repository serves as a baseline reproduction and learning exercise
to understand PINN fundamentals before extending the method to
nonlinear PDEs and hypersonic flow problems.

## Reference

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
Physics-informed neural networks: A deep learning framework for solving forward
and inverse problems involving nonlinear partial differential equations.
Journal of Computational Physics, 378, 686–707.
