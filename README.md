# SQP for Constrained Logistic Regression

## What is this project about?

**SQP (Sequential Quadratic Programming)** solves constrained optimization problems:

```
min f(x)   subject to   g(x) <= 0,   h(x) = 0
```

At each iteration, SQP replaces the hard problem with a simpler one (quadratic objective + linearized constraints), solves it, and takes a step.

**Application:** logistic regression where the regularization is an **explicit constraint** instead of a penalty:

```
min  J(w)  (average log-loss)     s.t.  ||w||_2^2 <= t   (ridge)   or   ||w||_1 <= t  (lasso, w = u - v, u,v >= 0)
```

Everything (QP solver, BFGS, merit function, line search) is implemented from scratch and validated against SciPy and scikit-learn.

## Files

| File | Role |
|---|---|
| `qp_solver.py` | QP subproblem solver: primal **active-set** method + **elastic mode** (handles infeasible linearizations, equalities and inequalities). |
| `sqp.py` | The SQP loop: equality + inequality constraints, damped BFGS (Powell) **or** exact Hessian, l1 merit function, Armijo line search, second-order correction (SOC), KKT stopping tests, per-iteration log. |
| `checks.py` | Finite-difference derivative checks and KKT residuals. |
| `problems.py` | Test problems: Rosenbrock on a disk, Hock-Schittkowski 6 and 7, and logistic regression with L2 / L1 constraints (with analytic gradients and Hessians). |
| `run_tests.py` | Validation: derivatives, toy problems vs SLSQP / trust-constr, edge cases, sklearn equivalence, lasso sparsity, convergence figure. |
| `run_experiments.py` | Regularization path, cross-validation for `t`, performance comparison, robustness (random starts, growing dimension, noise). |
| `figures/` | Output plots (`convergence.png`, `path.png`, `robustness.png`). |

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** (`sklearn.datasets.load_breast_cancer`): 569 samples, 30 numerical features, binary target (212 malignant, 357 benign). 70/30 stratified split, features standardized on the training set, unconstrained intercept. Robustness tests also use `make_classification` (noisy, correlated features, p up to 100).

## How to run

```bash
pip install -r requirements.txt
python run_tests.py          # validation + application (a few seconds)
python run_experiments.py    # path, CV, performance, robustness
```

## Using the SQP on your own problem

```python
from sqp import sqp
res = sqp(f, grad_f, x0, g=g, jac_g=jac_g, h=h, jac_h=jac_h, hess_L=None)  # hess_L optional (else BFGS)
res["x"], res["lam"], res["mu"], res["converged"], res["log"]
```

## Main results (this dataset)

- Same solution as SLSQP / trust-constr on all toy problems; KKT conditions satisfied to ~1e-10.
- Constrained ridge matches `sklearn.LogisticRegression` exactly (max weight difference < 1e-6) with `C = 1 / (2 n lambda*)`, where `lambda*` is the SQP multiplier.
- Lasso version gives sparse weights (e.g. 2 / 3 / 6 non-zeros out of 30 for t = 0.5 / 1.5 / 4).
- Exact Hessian: quadratic convergence (8-10 iterations). BFGS: fast linear convergence, not clearly superlinear (30-110 iterations).

## Limitations

- Active-set QP cost grows quickly with the number of constraints (the lasso form has 2p+1 inequalities).
- O(n^2) memory for the Hessian, O(n^3) per factorization: not meant for deep-learning-sized problems.
- Active-set QP can cycle on highly degenerate problems (iteration cap in place).
- On this small problem SLSQP (compiled Fortran) is faster in wall-clock time than this pure Python code.

## Reference

Nocedal & Wright, *Numerical Optimization*, Chapters 16 and 18.
