# NE362-PWR-Fuel-Thermal-Conduction-Model-Assignment

This repository contains a **Python-based** numerical simulation designed to solve steady-state heat conduction equations within a nuclear reactor core.

## Features
- **Discretization:** Converts 1D heat conduction equations into an $AT=b$ matrix system.
- **Boundary Conditions:** Compares the Centerplane Neumann symmetry using both Ghost-Point (Approach A) and Direct $\Delta x$ (Approach B) formulations.
- **Power Distributions:** Models both Uniform heat generation and Non-Uniform (Quadratic) power profiles to simulate thermal flux depression.
- **Mesh Study:** Features grid refinement tracking ($N = 5, 10, 20, 40, 80$) to evaluate convergence against analytical solutions.

## Environment & Dependencies
To run this simulation locally, you will need a **Python 3** environment (such as Spyder or VS Code) with the following standard engineering libraries installed:
- `numpy` (for matrix assembly and numerical operations)
- `scipy` (for the automated linear algebraic matrix solver)
- `matplotlib` (for plotting the grid-refinement convergence curves)
