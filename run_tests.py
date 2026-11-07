"""Validation (phases 2-3) et application ML (phase 4) du SQP.
Usage : python run_tests.py          (produit aussi figures/convergence.png)
"""
import os, time, warnings
import numpy as np
os.makedirs("figures", exist_ok=True)   # crée le dossier des figures
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sqp import sqp
from checks import check_derivative, kkt_residuals
from problems import (rosenbrock_disk, hs6, hs7, logistic_l2, logistic_l1, Problem)

np.random.seed(0)
ok = lambda b: "OK " if b else "ECHEC"


def scipy_solve(P, method):
    cons = []
    if len(P.g(P.x0)):
        cons.append({"type": "ineq", "fun": lambda x: -P.g(x), "jac": lambda x: -P.jac_g(x)})
    if len(P.h(P.x0)):
        cons.append({"type": "eq", "fun": P.h, "jac": P.jac_h})
    return minimize(P.f, P.x0, jac=P.grad_f, method=method, constraints=cons,
                    options={"maxiter": 1000})


# ------------------------------------------------------------------ 1. dérivées
print("=== 1. Vérification des dérivées (différences finies) ===")
for P in [rosenbrock_disk(), hs6(), hs7()]:
    x = P.x0 + 0.1
    e1 = check_derivative(P.f, P.grad_f, x)
    e2 = check_derivative(P.g, P.jac_g, x) if len(P.g(x)) else 0
    e3 = check_derivative(P.h, P.jac_h, x) if len(P.h(x)) else 0
    print(f"  {P.name:22s} grad f: {e1:.1e}  jac g: {e2:.1e}  jac h: {e3:.1e}  {ok(max(e1,e2,e3)<1e-5)}")

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                      random_state=0, stratify=data.target)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
for P in [logistic_l2(Xtr, ytr, 1.0), logistic_l1(Xtr, ytr, 2.0)]:
    x = np.random.randn(len(P.x0)) * 0.1 + (np.abs(P.x0) if "L1" in P.name else 0)
    e1 = check_derivative(P.f, P.grad_f, x)
    e2 = check_derivative(P.g, P.jac_g, x)
    # Hessien exact : différences finies du gradient de f (+ terme contraintes pour L2)
    H_fd = np.array([(P.grad_f(x + 1e-6 * e) - P.grad_f(x - 1e-6 * e)) / 2e-6 for e in np.eye(len(x))])
    H = P.hess_L(x, np.zeros(len(P.g(x))), np.zeros(0))
    e3 = np.max(np.abs(H_fd - H)) / (1 + np.max(np.abs(H_fd)))
    print(f"  {P.name:22s} grad f: {e1:.1e}  jac g: {e2:.1e}  hessien: {e3:.1e}  {ok(max(e1,e2,e3)<1e-5)}")

# ------------------------------------------------------------------ 2. problèmes jouets
print("\n=== 2. Problèmes jouets : SQP (BFGS / Hessien exact) vs SLSQP vs trust-constr ===")
refs = {"Rosenbrock-disque": np.array([0.7864151, 0.6176983]),
        "HS6 (égalité)": np.array([1.0, 1.0]), "HS7 (égalité)": np.array([0.0, np.sqrt(3)])}
for P in [rosenbrock_disk(), hs6(), hs7()]:
    for label, hl in [("SQP-BFGS ", None), ("SQP-exact", P.hess_L)]:
        r = sqp(**P.kwargs(), hess_L=hl)
        err = np.linalg.norm(r["x"] - refs[P.name])
        kk = kkt_residuals(P.grad_f, P.g, P.jac_g, P.h, P.jac_h, r["x"], r["lam"], r["mu"])
        print(f"  {P.name:20s} {label} iters={len(r['log']):3d} |x-x*|={err:.1e} "
              f"KKT(stat={kk['stationarity']:.0e}, prim={kk['primal']:.0e}, "
              f"compl={kk['complementarity']:.0e}) {ok(err<1e-5 and r['converged'])}")
    for m in ("SLSQP", "trust-constr"):
        s = scipy_solve(P, m)
        print(f"  {P.name:20s} {m:12s} nit={s.nit:3d} |x-x*|={np.linalg.norm(s.x-refs[P.name]):.1e}")

# ------------------------------------------------------------------ 3. cas limites
print("\n=== 3. Cas limites ===")
P = rosenbrock_disk(x0=(3.0, 3.0))                      # point initial infaisable
r = sqp(**P.kwargs()); print(f"  point initial infaisable (3,3) : converge={r['converged']} "
                              f"x={np.round(r['x'],5)} iters={len(r['log'])}  {ok(np.linalg.norm(r['x']-refs['Rosenbrock-disque'])<1e-5)}")
P = rosenbrock_disk(r2=10.0)                             # contrainte inactive
r = sqp(**P.kwargs()); print(f"  contrainte inactive (r2=10)    : x={np.round(r['x'],5)} lambda={r['lam'][0]:.1e}  "
                              f"{ok(np.linalg.norm(r['x']-1)<1e-5 and abs(r['lam'][0])<1e-6)}")
P = rosenbrock_disk(r2=1.0)                              # contrainte active
r = sqp(**P.kwargs()); print(f"  contrainte active  (r2=1)      : g(x*)={P.g(r['x'])[0]:.1e} lambda={r['lam'][0]:.3f}  {ok(r['lam'][0]>0)}")
kappa = 1e6                                              # mauvais conditionnement
Q = np.diag([1.0, kappa])
Pc = Problem(lambda x: 0.5 * x @ Q @ x, lambda x: Q @ x, [1.0, 1.0],
             h=lambda x: np.array([x[0] + x[1] - 1]), jac_h=lambda x: np.array([[1.0, 1.0]]),
             name="quadratique cond=1e6")
xs = np.array([kappa, 1.0]) / (1 + kappa)               # solution analytique
for label, hl in [("BFGS ", None), ("exact", lambda x, l, m: Q)]:
    r = sqp(**Pc.kwargs(), hess_L=hl)
    print(f"  mauvais conditionnement (1e6)  : {label} iters={len(r['log']):3d} converge={r['converged']} "
          f"|x-x*|={np.linalg.norm(r['x']-xs):.1e}  {ok(np.linalg.norm(r['x']-xs)<1e-6)}")

# ------------------------------------------------------------------ 4. application ML (L2)
print("\n=== 4. Régression logistique avec contrainte ridge ||w||^2 <= t ===")
n = len(ytr)
for t in (0.5, 2.0, 8.0):
    P = logistic_l2(Xtr, ytr, t)
    t0 = time.time(); r = sqp(**P.kwargs(), hess_L=P.hess_L); dt = time.time() - t0
    w, lam = r["x"][1:], r["lam"][0]
    C = 1.0 / (2.0 * n * lam)                            # équivalence contrainte <-> pénalité
    sk = LogisticRegression(C=C, tol=1e-12, max_iter=20000).fit(Xtr, ytr)
    diff = max(np.max(np.abs(w - sk.coef_[0])), abs(r["x"][0] - sk.intercept_[0]))
    acc_te = np.mean(((Xte @ w + r["x"][0]) > 0) == yte)
    print(f"  t={t:4.1f}  iters={len(r['log']):2d} ({dt:.2f}s) ||w||^2={w@w:.4f} lambda={lam:.4f} "
          f"-> C_sklearn={C:.3f}  max|w_sqp-w_sk|={diff:.1e}  acc_test={acc_te:.3f}  {ok(diff<1e-4)}")

# ------------------------------------------------------------------ 5. application ML (L1)
print("\n=== 5. Version lasso ||w||_1 <= t  (w = u - v, u,v >= 0) ===")
for t in (0.5, 1.5, 4.0):
    P = logistic_l1(Xtr, ytr, t)
    t0 = time.time(); r = sqp(**P.kwargs(), hess_L=P.hess_L); dt = time.time() - t0
    p = Xtr.shape[1]; w = r["x"][1:p + 1] - r["x"][p + 1:]
    acc_te = np.mean(((Xte @ w + r["x"][0]) > 0) == yte)
    sk = LogisticRegression(penalty="l1", solver="liblinear",
                            C=1.0 / (r["lam"][0] * n) if r["lam"][0] > 1e-9 else 1e3, tol=1e-10).fit(Xtr, ytr)
    print(f"  t={t:3.1f} conv={r['converged']} iters={len(r['log']):3d} ({dt:.1f}s) ||w||_1={np.abs(w).sum():.3f} "
          f"non-nuls={np.sum(np.abs(w)>1e-6):2d}/{p} acc_test={acc_te:.3f} | sklearn L1 (indicatif) non-nuls={np.sum(np.abs(sk.coef_[0])>1e-6)}")

# ------------------------------------------------------------------ 6. courbe de convergence
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = logistic_l2(Xtr, ytr, 1.0)
runs = {"SQP BFGS (Powell)": sqp(**P.kwargs(), tol=1e-9),
        "SQP Hessien exact": sqp(**P.kwargs(), hess_L=P.hess_L, tol=1e-9)}
fstar = runs["SQP Hessien exact"]["log"][-1]["f"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for name, r in runs.items():
    ks = [l["k"] for l in r["log"]]
    ax[0].semilogy(ks, [max(abs(l["f"] - fstar), 1e-16) for l in r["log"]], "o-", label=name)
    ax[1].semilogy(ks, [max(l["viol"], 1e-16) for l in r["log"]], "o-", label=name)
    ax[2].semilogy(ks, [max(l["kkt"], 1e-16) for l in r["log"]], "o-", label=name)
ax[0].set(title="|f - f*|  (échelle log)", xlabel="itération")
ax[1].set(title="violation de contrainte", xlabel="itération")
ax[2].set(title="résidu KKT ||grad L||_inf", xlabel="itération")
ax[0].legend(); [a.grid(alpha=.3) for a in ax]; plt.tight_layout(); plt.savefig("figures/convergence.png", dpi=150)
print("\nFigure sauvegardée : figures/convergence.png")
for name, r in runs.items():
    print(f"  {name:20s} iters={len(r['log'])}  dernières KKT: {[f'{l['kkt']:.0e}' for l in r['log'][-5:]]}")