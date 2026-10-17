# NE362-PWR-Fuel-Thermal-Conduction-Model-Assignment

A 1D steady-state thermal conduction model in Python simulating the radial temperature profile within a PWR nuclear fuel pellet using the finite-difference method.

## Features
- **Discretization:** Converts 1D heat conduction equations into an $AT=b$ matrix system.
- **Boundary Conditions:** Compares the Centerplane Neumann symmetry using both Ghost-Point (Approach A) and Direct $\Delta x$ (Approach B) formulations.
- **Power Distributions:** Models both Uniform heat generation and Non-Uniform (Quadratic) power profiles to simulate thermal flux depression.
- **Mesh Study:** Features grid refinement tracking ($N = 5, 10, 20, 40, 80$) to evaluate convergence against analytical solutions.
