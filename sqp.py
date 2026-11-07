"""SQP complet : contraintes d'égalité ET d'inégalité.

    min f(x)   s.t.   g(x) <= 0,   h(x) = 0

Ingrédients (Nocedal & Wright, chap. 18) :
  - sous-problème QP à chaque itération (qp_solver.solve_qp, mode élastique)
  - Hessien du lagrangien : BFGS avec damping de Powell (éq. 18.14-18.15)
                            OU Hessien exact rendu défini positif (option hess_L)
  - globalisation : fonction de mérite l1  phi = f + pen*(sum max(0,g) + sum |h|)
                    + recherche linéaire d'Armijo + correction de second ordre (SOC)
"""
import time
import numpy as np
from qp_solver import solve_qp


def powell_damped_bfgs(B, s, y, theta_min=0.2):
    """Mise à jour BFGS amortie : garde B symétrique définie positive."""
    Bs = B @ s
    sBs = s @ Bs
    sy = s @ y
    theta = 1.0 if sy >= theta_min * sBs else (1 - theta_min) * sBs / (sBs - sy)
    r = theta * y + (1 - theta) * Bs
    return B - np.outer(Bs, Bs) / sBs + np.outer(r, r) / (s @ r)


def make_pd(H, delta=1e-6):
    """Rend une matrice symétrique définie positive (coupe les valeurs propres)."""
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, delta * max(1.0, np.abs(w).max()))
    return (V * w) @ V.T


def _l1_viol(g, h):
    return np.sum(np.maximum(g, 0.0)) + np.sum(np.abs(h))


def sqp(f, grad_f, x0, g=None, jac_g=None, h=None, jac_h=None, hess_L=None,
        tol=1e-8, max_iter=200, eta=1e-4, soc=True, verbose=False):
    """Résout le problème par SQP.

    hess_L(x, lam, mu) -> Hessien exact du lagrangien (optionnel ; sinon BFGS).
    Retourne un dict : x, lam (inégalités), mu (égalités), converged, log (liste de dicts).
    """
    x = np.asarray(x0, float).copy()
    n = x.size
    g = g or (lambda z: np.zeros(0))
    jac_g = jac_g or (lambda z: np.zeros((0, n)))
    h = h or (lambda z: np.zeros(0))
    jac_h = jac_h or (lambda z: np.zeros((0, n)))

    B = np.eye(n)
    pen = 1.0
    lam, mu = np.zeros(len(g(x))), np.zeros(len(h(x)))
    log, converged, t0 = [], False, time.time()

    def grad_L(z, l, m):
        return grad_f(z) + jac_g(z).T @ l + jac_h(z).T @ m

    for k in range(max_iter):
        fx, gf = f(x), grad_f(x)
        gx, Jg, hx, Jh = g(x), jac_g(x), h(x), jac_h(x)
        if hess_L is not None:
            # Hessien exact rendu PD + régularisation tau*I qui -> 0 près de la solution
            # (tau ~ ||grad L||) : évite les pas géants quand la courbure est ~0 loin de x*.
            tau = min(1.0, np.linalg.norm(grad_L(x, lam, mu), np.inf))
            B = make_pd(hess_L(x, lam, mu)) + tau * np.eye(n)

        # ---- sous-problème QP ----
        d, lam_n, mu_n, qp_it, gap = solve_qp(B, gf, Jh, hx, Jg, gx)

        # ---- test d'arrêt (KKT) ----
        gL = gf + Jg.T @ lam_n + Jh.T @ mu_n
        viol = max(np.max(np.maximum(gx, 0.0), initial=0.0), np.max(np.abs(hx), initial=0.0))
        compl = np.max(np.abs(lam_n * gx), initial=0.0)
        rec = dict(k=k, f=fx, viol=viol, kkt=np.linalg.norm(gL, np.inf),
                   d=np.linalg.norm(d, np.inf), alpha=np.nan, pen=pen,
                   qp_iters=qp_it, time=time.time() - t0)
        log.append(rec)
        if verbose:
            print(f"{k:3d} f={fx:+.8e} viol={viol:.1e} kkt={rec['kkt']:.1e} |d|={rec['d']:.1e}")
        small_step = rec["d"] <= 1e-10 * (1.0 + np.linalg.norm(x, np.inf))   # utile si mal conditionné
        if (rec["kkt"] <= tol or small_step) and viol <= tol and compl <= tol:
            lam, mu, converged = lam_n, mu_n, True
            break

        # ---- fonction de mérite l1 : mise à jour de la pénalité ----
        v0 = _l1_viol(gx, hx)
        v_lin = _l1_viol(gx + Jg @ d, hx + Jh @ d)
        mults = np.concatenate([np.abs(lam_n), np.abs(mu_n)])
        if mults.size:
            pen = max(pen, 1.1 * mults.max())
        if v0 - v_lin > 1e-12:
            pen = max(pen, (gf @ d + 0.5 * d @ B @ d) / (0.5 * (v0 - v_lin)))
        D = gf @ d - pen * (v0 - v_lin)                # dérivée directionnelle de phi
        phi = lambda z: f(z) + pen * _l1_viol(g(z), h(z))
        phi0 = phi(x)
        slack = 1e-14 * max(1.0, abs(phi0))          # tolérance d'arrondi pour Armijo
        rec["pen"] = pen

        if D >= 0.0:                                # pas de descente : on repart de B = I
            B = np.eye(n)
            if verbose:
                print("   direction non descendante -> reset B")
            lam, mu = lam_n, mu_n
            continue

        # ---- recherche linéaire d'Armijo (+ SOC si le pas complet est refusé) ----
        step, alpha = d, 1.0
        if phi(x + d) > phi0 + eta * D + slack and soc and (len(gx) or len(hx)):
            g2, h2 = g(x + d) - Jg @ d, h(x + d) - Jh @ d
            d_soc, *_ = solve_qp(B, gf, Jh, h2, Jg, g2)
            if phi(x + d_soc) <= phi0 + eta * D + slack:
                step = d_soc
        if step is d:
            while phi(x + alpha * d) > phi0 + eta * alpha * D + slack and alpha > 1e-14:
                alpha *= 0.5
            if alpha <= 1e-14:
                if verbose:
                    print("   recherche linéaire échouée")
                lam, mu = lam_n, mu_n
                break
        rec["alpha"] = alpha

        # ---- mise à jour de x et de B ----
        x_new = x + alpha * step
        if hess_L is None:
            s = x_new - x
            yv = grad_L(x_new, lam_n, mu_n) - grad_L(x, lam_n, mu_n)
            if s @ s > 1e-30:
                B = powell_damped_bfgs(B, s, yv)
        x, lam, mu = x_new, lam_n, mu_n

    return dict(x=x, lam=lam, mu=mu, converged=converged, log=log)
