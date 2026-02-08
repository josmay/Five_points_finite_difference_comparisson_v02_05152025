# Five-Point Finite Difference Operator and Forward Wave Model  
*Supplementary Material*

## Overview

This repository provides the **supplementary numerical material** for the article:

> **[Title of your article]**  
> *[Journal name, year]*

It contains the Python implementation of:

- A **five-point finite difference operator** for high-order spatial derivatives
- A **one-dimensional forward wave propagation solver**
- Numerical tools used to evaluate and validate the methods presented in the paper

The code is intended to support **reproducibility**, **methodological transparency**, and **independent verification** of the numerical results discussed in the manuscript.

---

## Scope of the Repository

This repository focuses exclusively on the **numerical components** of the work:

- Construction of the five-point finite difference operator \( \mathbf{D}_5 \)
- Treatment of boundary stencils for first- through fourth-order derivatives
- Forward modeling of the 1D wave equation used in the inverse problems
- Comparison with classical finite difference schemes

All theoretical derivations, proofs, and statistical modeling are presented in the main article.  
This repository **does not** include the inference algorithms or optimization routines discussed in the paper.

---

## Numerical Methods Implemented

### Five-Point Finite Difference Operator

The five-point operator is constructed by solving local linear systems derived from Taylor expansions.  
Different stencils are used depending on the node location:

- Left boundary
- Near-left boundary
- Interior nodes
- Near-right boundary
- Right boundary

Each stencil yields a vector of coefficients \( \boldsymbol{\alpha}_i \), which are assembled into a global sparse operator matrix.

This implementation supports derivatives of order:

\[
n = 1, 2, 3, 4
\]

and ensures consistent accuracy across the domain, including at the boundaries.

---

### Forward Wave Model

The forward model solves the one-dimensional wave equation using:

- Explicit time integration
- Spatial discretization via the five-point operator
- Open (absorbing) boundary conditions

The solver is designed to match the forward operators used in the Bayesian inversion framework presented in the article.

---

## Repository Structure

