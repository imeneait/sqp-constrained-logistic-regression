"""Sous-solveur QP : active-set primal + mode élastique (tolère les QP infaisables).

Problème de l'itération SQP (B définie positive) :
    min_d  1/2 d^T B d + c^T d
    s.t.   A_eq d + h  = 0
           A_in d + g <= 0

Mode élastique (Nocedal & Wright, sect. 18.3) : on relâche les contraintes avec des
variables d'écart pénalisées en l1 ; le QP relâché est TOUJOURS faisable, on peut donc
le résoudre par un active-set primal démarré en d = 0. Si le QP original est faisable
et rho est assez grand, les écarts sont nuls à la solution (pénalité exacte).
"""
import numpy as np


# ----------------------------------------------------------------------------
# Active-set primal (NW, Algorithme 16.3) pour un QP strictement convexe
#     min 1/2 z'Hz + g'z   s.t.  E z = e,  C z >= r,   à partir d'un z0 faisable
# ----------------------------------------------------------------------------
def active_set_qp(H, g, E, e, C, r, z0, max_iter=1000, tol=1e-14):
    n = H.shape[0]
    me, mi = E.shape[0], C.shape[0]
    z = z0.astype(float).copy()

    # ensemble de travail initial : contraintes actives et linéairement indépendantes
    W = []
    rows = E.copy()
    for i in range(mi):
        if abs(C[i] @ z - r[i]) <= tol * (1.0 + abs(r[i])):
            cand = np.vstack([rows, C[i]])
            if np.linalg.matrix_rank(cand, tol=1e-9) == cand.shape[0]:
                W.append(i)
                rows = cand

    y = np.zeros(me)
    for it in range(max_iter):
        Aw = np.vstack([E, C[W]]) if W else E
        k = Aw.shape[0]
        q = H @ z + g
        K = np.block([[H, Aw.T], [Aw, np.zeros((k, k))]])
        rhs = np.concatenate([-q, np.zeros(k)])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        p, y = sol[:n], sol[n:]

        if np.linalg.norm(p, np.inf) <= 1e-13 * (1.0 + np.linalg.norm(z, np.inf)):
            if k > 0:                            # multiplicateurs plus précis : moindres carrés
                y = np.linalg.lstsq(Aw.T, -q, rcond=None)[0]   # sur Aw seul (bien conditionné)
            lam_W = -y[me:]                      # multiplicateurs des C z >= r
            if len(W) == 0 or lam_W.min() >= -1e-10:
                lam = np.zeros(mi)
                if W:
                    lam[W] = lam_W
                return z, lam, y[:me], it + 1
            W.pop(int(np.argmin(lam_W)))          # on relâche la pire contrainte
        else:
            alpha, block = 1.0, None
            for i in range(mi):
                if i in W:
                    continue
                ap = C[i] @ p
                if ap < -1e-12:
                    a = max((r[i] - C[i] @ z) / ap, 0.0)
                    if a < alpha:
                        alpha, block = a, i
            z = z + alpha * p
            if block is not None:
                cand = np.vstack([Aw, C[block]])
                if np.linalg.matrix_rank(cand, tol=1e-9) == cand.shape[0]:
                    W.append(block)
    raise RuntimeError("active_set_qp : nombre max d'itérations atteint")


# ----------------------------------------------------------------------------
# QP de l'itération SQP, avec relaxation élastique
# ----------------------------------------------------------------------------
def solve_qp(B, c, A_eq, h, A_in, g, rho=1e3, eps=1e-3):
    """Renvoie (d, lam_in, mu_eq, n_iter_actset, max_ecart).

    Convention des multiplicateurs : grad f + A_in^T lam + A_eq^T mu = B-terme = 0,
    avec lam >= 0.
    """
    n, me, mi = B.shape[0], len(h), len(g)
    if me == 0 and mi == 0:
        return -np.linalg.solve(B, c), np.zeros(0), np.zeros(0), 1, 0.0

    for _ in range(4):                       # on augmente rho si l'écart n'est pas nul
        N = n + 2 * me + mi
        H = np.zeros((N, N)); H[:n, :n] = B; H[n:, n:] = eps * np.eye(N - n)
        gv = np.concatenate([c, rho * np.ones(N - n)])
        ip, iq, is_ = n, n + me, n + 2 * me          # début des blocs p, q, s

        # égalités :  A_eq d - p + q = -h
        E = np.zeros((me, N))
        if me:
            E[:, :n] = A_eq
            E[:, ip:ip + me] = -np.eye(me)
            E[:, iq:iq + me] = np.eye(me)
        e = -np.asarray(h, float)

        # inégalités (forme >=) :  -A_in d + s >= g ;  p,q,s >= 0
        C = np.zeros((mi + N - n, N)); r = np.zeros(mi + N - n)
        if mi:
            C[:mi, :n] = -A_in
            C[:mi, is_:is_ + mi] = np.eye(mi)
            r[:mi] = g
        C[mi:, n:] = np.eye(N - n)

        z0 = np.zeros(N)                                   # d = 0 + écarts faisables
        z0[ip:ip + me] = np.maximum(h, 0.0)
        z0[iq:iq + me] = np.maximum(-np.asarray(h, float), 0.0)
        z0[is_:is_ + mi] = np.maximum(g, 0.0)

        z, lam_all, nu, nit = active_set_qp(H, gv, E, e, C, r, z0)
        gap = z[n:].max() if N > n else 0.0
        if gap < 1e-8 or (gap < 1e-6 and rho >= 1e6):
            break
        rho *= 10
    d = z[:n]
    return d, lam_all[:mi], nu, nit, gap
