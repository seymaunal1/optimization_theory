# Optimization Theory: Homework 2

This project was developed as part of an undergraduate course in Optimization (EE457).

This repository contains the implementation and analysis of Homework 2 in Optimization Theory.  
It shows the application of different optimization algorithms and compares their performance in terms of convergence and efficiency.

## Methods

The algorithms examined in this project are:

1. Gradient descent with fixed step sizes  
2. Steepest descent with golden section line search  
3. Gradient descent with Armijo-Goldstein line search  
4. Newton’s method  
5. Modified Newton’s method with golden section line search  
6. Modified Newton’s method with Armijo-Goldstein line search  

## Files

- `UnalAydemirEE457hw2.m` - main MATLAB code  
- `UnalAydemirEE457hw2.pdf` - project report
- `EE__45701.pdf` - syllabus of the course

## Problem Setup

The Powell function is defined as:

f(x) = (x₁ + 10x₂)² + 5(x₃ − x₄)² + (x₂ − 2x₃)⁴ + 10(x₁ − x₄)⁴  

The initial point is:

x(0) = [3, −1, 0, 1]ᵀ  

The stopping criterion is:

‖∇f(x)‖ ≤ 10⁻²  

These conditions are used for all algorithms.
