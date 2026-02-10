# Overview

This repository provides the supplementary numerical material for the article:

Greedy Stein Variational Gradient Descent for Inverse Wave Propagation Problems
Revista de Ingeniería de la UNAM, 2026

# The repository is organized into two main components:

- Source Code

- Supplementary Material (PDF)

This material supports the numerical experiments and methodological developments presented in the associated journal article.

# Repository Structure


├── Source_Code/
│ ├── cuda_modelo_directo_XX_open_boundary.py
│ ├── cuda_soporte.py
│ ├── FD_solucion_onda_XX.py
│ ├── finite_difference_operators.py
│ └── gpu_bSplines.py
│
├── Supplementary_Material/
│ └── five_point_operator_derivation.pdf
│
└── README.md

# Source Code

The Source Code directory contains Python implementations for solving the wave prospection problem in stratified media.

- cuda_modelo_directo_XX_open_boundary.py
Implements the forward wave solver using CUDA-enabled GPUs and open boundary conditions.

- cuda_soporte.py
Provides auxiliary routines for numerical domain construction and GPU-based computations.

- FD_solucion_onda_XX.py
Implements the forward wave model using different finite difference operators.

- finite_difference_operators.py
Constructs five-point finite difference operators for arbitrary spatial discretization steps.

- gpu_bSplines.py
Implements β-spline parameterizations of the velocity field for low-contrast stratified media.

## Supplementary Material (PDF)

The Supplementary_Material directory contains a LaTeX-based document describing:

- The derivation of the five-point finite difference operator,

- A detailed error analysis of the approximation,

- The formulation of the forward wave propagation model.

This document complements the numerical results presented in the main article and provides full methodological transparency.

# Numerical Methods Implemented

The forward wave model is solved using finite difference schemes combined with explicit time stepping, where the solution at time $𝑡_{𝑛+1} depends directly on the solution at time $t_{n}$.

To accelerate computations—particularly when solving multiple independent models simultaneously—we leverage GPU acceleration using the CuPy library. CuPy mirrors NumPy syntax, allowing efficient GPU–CPU interoperability with minimal code changes.

# Five-Point Finite Difference Operator

To assess the numerical behavior of the five-point finite difference operator, we first apply it to the approximation of derivatives of known analytical functions.

We then incorporate the operator into the forward wave model and analyze the numerical behavior near domain boundaries. Since exact analytical solutions are generally unavailable at the boundaries, we use the standard deviation across multiple spatial resolutions as an error metric.

As the spatial resolution increases, all finite difference solutions converge toward a common (unknown) limit. The standard deviation therefore provides a quantitative measure of the uncertainty associated with the numerical approximation.

# How to Run the Code
## Requirements

- Python ≥ 3.9

- NumPy

- CuPy (for GPU execution)

- CUDA-compatible GPU (optional but recommended)

## Installation

CPU-only dependencies:

- pip install numpy

GPU support (example for CUDA 12.x):

- pip install cupy-cuda12x

## Running a Forward Model

GPU-based forward solver:

- python cuda_modelo_directo_01_open_boundary.py

CPU-based finite difference solver:

- python FD_solucion_onda_01.py

Model parameters (spatial resolution, time step, velocity model, etc.) are defined directly in each script and correspond to the configurations used in the associated article.

## Reproducibility and Journal Compliance

- All numerical solvers correspond exactly to those used in the published experiments.

- No post-processing or parameter tuning was performed outside the scripts provided here.

- The repository is intended for reproducibility, validation, and methodological transparency, in line with journal supplementary material standards.

# Citation

If you use this repository, please cite it as:

BibTeX

- @misc{VaronaSantana2025FivePointFD,
author = {Varona Santana, Jose Luis},
title = {Five-Point Finite Difference Operator and Forward Wave Model},
year = {2025},
howpublished = {https://github.com/josmay/Five_points_finite_difference_comparisson_v02_05152025}
,
note = {Supplementary numerical material for the associated journal article}
}