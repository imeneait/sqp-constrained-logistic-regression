"""Outils de vérification : gradients par différences finies et résidus KKT."""
import numpy as np


def fd_gradient(f, x, eps=1e-6):
    """Gradient (ou Jacobienne si f est vectorielle) par différences finies centrées."""
    x = np.asarray(x, float)
    cols = []
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = eps
        cols.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * eps))
    J = np.array(cols).T
    return J[0] if J.shape[0] == 1 and np.ndim(f(x)) == 0 else J


def check_derivative(fun, deriv, x, eps=1e-6):
    """Erreur relative max entre dérivée analytique et différences finies."""
    num, ana = np.asarray(fd_gradient(fun, x, eps)), np.asarray(deriv(x))
    num = num.reshape(ana.shape)
    return np.max(np.abs(num - ana)) / (1.0 + np.max(np.abs(num)))


def kkt_residuals(grad_f, g, jac_g, h, jac_h, x, lam, mu):
    """Résidus KKT : stationnarité, faisabilité primale, duale, complémentarité."""
    gx, hx = g(x), h(x)
    gL = grad_f(x) + jac_g(x).T @ lam + jac_h(x).T @ mu
    return dict(
        stationarity=np.linalg.norm(gL, np.inf),
        primal=max(np.max(np.maximum(gx, 0), initial=0.0), np.max(np.abs(hx), initial=0.0)),
        dual=max(0.0, -np.min(lam, initial=0.0)),
        complementarity=np.max(np.abs(lam * gx), initial=0.0))
