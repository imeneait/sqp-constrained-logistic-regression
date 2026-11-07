"""Définitions de problèmes : jouets (Rosenbrock, Hock-Schittkowski) et logistique contrainte."""
import numpy as np


class Problem:
    """Conteneur : f, grad_f, g, jac_g, h, jac_h, hess_L (optionnel), x0."""
    def __init__(self, f, grad_f, x0, g=None, jac_g=None, h=None, jac_h=None, hess_L=None, name=""):
        n = len(x0)
        self.f, self.grad_f, self.x0, self.name = f, grad_f, np.asarray(x0, float), name
        self.g = g or (lambda x: np.zeros(0))
        self.jac_g = jac_g or (lambda x: np.zeros((0, n)))
        self.h = h or (lambda x: np.zeros(0))
        self.jac_h = jac_h or (lambda x: np.zeros((0, n)))
        self.hess_L = hess_L

    def kwargs(self):
        return dict(f=self.f, grad_f=self.grad_f, x0=self.x0, g=self.g, jac_g=self.jac_g,
                    h=self.h, jac_h=self.jac_h)


# ---------------- problèmes jouets à solution connue ----------------
def rosenbrock_disk(r2=1.0, x0=(-1.0, 1.0)):
    """Rosenbrock sous contrainte disque x^2+y^2 <= r2. Pour r2=1 : x*~(0.7864, 0.6177)."""
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    gr = lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                             200 * (x[1] - x[0] ** 2)])
    g = lambda x: np.array([x[0] ** 2 + x[1] ** 2 - r2])
    jg = lambda x: np.array([[2 * x[0], 2 * x[1]]])
    def hl(x, lam, mu):
        H = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]])
        return H + lam[0] * 2 * np.eye(2)
    return Problem(f, gr, x0, g=g, jac_g=jg, hess_L=hl, name="Rosenbrock-disque")


def hs6(x0=(-1.2, 1.0)):
    """HS6 : min (1-x1)^2  s.t. 10(x2 - x1^2) = 0.  Solution (1, 1), f* = 0."""
    f = lambda x: (1 - x[0]) ** 2
    gr = lambda x: np.array([-2 * (1 - x[0]), 0.0])
    h = lambda x: np.array([10 * (x[1] - x[0] ** 2)])
    jh = lambda x: np.array([[-20 * x[0], 10.0]])
    def hl(x, lam, mu):
        return np.array([[2.0, 0.0], [0.0, 0.0]]) + mu[0] * np.array([[-20.0, 0.0], [0.0, 0.0]])
    return Problem(f, gr, x0, h=h, jac_h=jh, hess_L=hl, name="HS6 (égalité)")


def hs7(x0=(2.0, 2.0)):
    """HS7 : min ln(1+x1^2) - x2  s.t. (1+x1^2)^2 + x2^2 - 4 = 0. Solution (0, sqrt3), f* = -sqrt3."""
    f = lambda x: np.log(1 + x[0] ** 2) - x[1]
    gr = lambda x: np.array([2 * x[0] / (1 + x[0] ** 2), -1.0])
    h = lambda x: np.array([(1 + x[0] ** 2) ** 2 + x[1] ** 2 - 4])
    jh = lambda x: np.array([[4 * x[0] * (1 + x[0] ** 2), 2 * x[1]]])
    def hl(x, lam, mu):
        a = 1 + x[0] ** 2
        Hf = np.array([[2 * (1 - x[0] ** 2) / a ** 2, 0.0], [0.0, 0.0]])
        Hh = np.array([[4 * a + 8 * x[0] ** 2, 0.0], [0.0, 2.0]])
        return Hf + mu[0] * Hh
    return Problem(f, gr, x0, h=h, jac_h=jh, hess_L=hl, name="HS7 (égalité)")


# ---------------- régression logistique contrainte ----------------
def _loss_parts(X, y):
    def f_w(b, w):
        z = b + X @ w
        return np.mean(np.logaddexp(0, z) - y * z)
    def grads(b, w):
        z = b + X @ w
        r = (1 / (1 + np.exp(-z)) - y) / len(y)
        return r.sum(), X.T @ r
    def hess(b, w):
        z = b + X @ w
        p = 1 / (1 + np.exp(-z))
        D = p * (1 - p) / len(y)
        Xa = np.hstack([np.ones((len(y), 1)), X])
        return Xa.T @ (Xa * D[:, None])
    return f_w, grads, hess


def logistic_l2(X, y, t):
    """min log-loss moyenne  s.t.  ||w||_2^2 <= t   (intercept b libre). Variables x = [b, w]."""
    f_w, grads, hess = _loss_parts(X, y)
    p = X.shape[1]
    f = lambda x: f_w(x[0], x[1:])
    def gr(x):
        gb, gw = grads(x[0], x[1:]); return np.concatenate([[gb], gw])
    g = lambda x: np.array([x[1:] @ x[1:] - t])
    jg = lambda x: np.concatenate([[0.0], 2 * x[1:]])[None, :]
    def hl(x, lam, mu):
        return hess(x[0], x[1:]) + lam[0] * 2 * np.diag(np.r_[0.0, np.ones(p)])
    x0 = np.zeros(p + 1); x0[1:] = 1e-2
    return Problem(f, gr, x0, g=g, jac_g=jg, hess_L=hl, name=f"logreg-L2 (t={t})")


def logistic_l1(X, y, t):
    """min log-loss moyenne  s.t.  ||w||_1 <= t, avec w = u - v, u,v >= 0.  x = [b, u, v]."""
    f_w, grads, hess = _loss_parts(X, y)
    p = X.shape[1]
    T = np.zeros((p + 1, 2 * p + 1)); T[0, 0] = 1
    T[1:, 1:p + 1] = np.eye(p); T[1:, p + 1:] = -np.eye(p)   # [b,w] = T [b,u,v]
    f = lambda x: f_w(x[0], x[1:p + 1] - x[p + 1:])
    def gr(x):
        gb, gw = grads(x[0], x[1:p + 1] - x[p + 1:])
        return T.T @ np.concatenate([[gb], gw])
    # contraintes : sum(u+v) - t <= 0 ; -u <= 0 ; -v <= 0   (toutes linéaires)
    G = np.zeros((1 + 2 * p, 2 * p + 1))
    G[0, 1:] = 1.0
    G[1:, 1:] = -np.eye(2 * p)
    gvec = np.concatenate([[-t], np.zeros(2 * p)])
    g = lambda x: G @ x + gvec
    jg = lambda x: G
    hl = lambda x, lam, mu: T.T @ hess(x[0], x[1:p + 1] - x[p + 1:]) @ T
    x0 = np.zeros(2 * p + 1); x0[1:] = t / (4 * p)
    return Problem(f, gr, x0, g=g, jac_g=jg, hess_L=hl, name=f"logreg-L1 (t={t})")
