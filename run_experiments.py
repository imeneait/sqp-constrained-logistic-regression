"""Phases 4-5 : chemin de régularisation, choix de t par validation croisée,
performances (vs SLSQP / trust-constr / steepest descent à pénalité), robustesse.
Usage : python run_experiments.py   -> figures/path.png, figures/robustness.png
"""
import os, time, warnings
import numpy as np
os.makedirs("figures", exist_ok=True)   # crée le dossier des figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sqp import sqp
from problems import logistic_l2, logistic_l1

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

def predict_proba(x, X):
    return 1 / (1 + np.exp(-(x[0] + X @ x[1:])))

def fit_l2(X, y, t, **kw):
    P = logistic_l2(X, y, t)
    return sqp(**P.kwargs(), hess_L=P.hess_L, **kw), P

# ---------------------------------------------------------------- chemin de régularisation
print("=== Chemin de régularisation (L2 contraint) ===")
ts = np.logspace(-1.5, 2.3, 18)
rows = []
for t in ts:
    r, P = fit_l2(Xtr, ytr, t)
    w = r["x"]
    ptr, pte = predict_proba(w, Xtr), predict_proba(w, Xte)
    rows.append(dict(t=t, norm=np.linalg.norm(w[1:]), acc_tr=accuracy_score(ytr, ptr > .5),
                     acc_te=accuracy_score(yte, pte > .5), ll_tr=log_loss(ytr, ptr), ll_te=log_loss(yte, pte),
                     iters=len(r["log"]), conv=r["converged"]))
print(f"  convergence : {sum(r['conv'] for r in rows)}/{len(rows)} valeurs de t")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogx(ts, [r["norm"] for r in rows], "o-"); ax[0].set(xlabel="t", title="||w||_2 (chemin de régularisation)")
ax[1].semilogx(ts, [r["acc_tr"] for r in rows], "o-", label="train")
ax[1].semilogx(ts, [r["acc_te"] for r in rows], "s-", label="test"); ax[1].set(xlabel="t", title="accuracy"); ax[1].legend()
ax[2].semilogx(ts, [r["ll_tr"] for r in rows], "o-", label="train")
ax[2].semilogx(ts, [r["ll_te"] for r in rows], "s-", label="test"); ax[2].set(xlabel="t", title="log-loss (biais-variance)"); ax[2].legend()
[a.grid(alpha=.3) for a in ax]; plt.tight_layout(); plt.savefig("figures/path.png", dpi=150)

# ---------------------------------------------------------------- choix de t par validation croisée
print("\n=== Choix de t par validation croisée (5 plis, log-loss) ===")
cv_scores = []
for t in ts:
    s = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
        sc2 = StandardScaler().fit(Xtr[a])
        r, _ = fit_l2(sc2.transform(Xtr[a]), ytr[a], t)
        s.append(log_loss(ytr[b], predict_proba(r["x"], sc2.transform(Xtr[b]))))
    cv_scores.append(np.mean(s))
t_best = ts[int(np.argmin(cv_scores))]
r, P = fit_l2(Xtr, ytr, t_best); pte = predict_proba(r["x"], Xte)
print(f"  t* = {t_best:.3f}")
print(f"  test : accuracy={accuracy_score(yte, pte>.5):.3f}  F1={f1_score(yte, pte>.5):.3f}  log-loss={log_loss(yte, pte):.3f}")
print("  matrice de confusion (test) :\n", confusion_matrix(yte, pte > .5))

# ---------------------------------------------------------------- performances
print("\n=== Performances (t = t*) ===")
P = logistic_l2(Xtr, ytr, t_best)
def timeit(fn, n=5):
    ts_ = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts_.append(time.perf_counter() - t0)
    return out, np.median(ts_)
cons = [{"type": "ineq", "fun": lambda x: -P.g(x), "jac": lambda x: -P.jac_g(x)}]
def scipy_run(m): return minimize(P.f, P.x0, jac=P.grad_f, method=m, constraints=cons, options={"maxiter": 2000})
def penalty_gd(mu=100.0, iters=20000, lr=None):
    """Baseline faible : plus forte pente sur f + mu*max(0,g)^2 (gradient à pas fixe)."""
    x = P.x0.copy()
    for k in range(iters):
        gx = P.g(x)[0]
        gr = P.grad_f(x) + 2 * mu * max(gx, 0) * P.jac_g(x)[0]
        x -= 0.1 * gr
    return x
fs = {}
(r_b, dt_b) = timeit(lambda: sqp(**P.kwargs(), tol=1e-9)); fs["SQP-BFGS"] = (len(r_b["log"]), dt_b, P.f(r_b["x"]), max(P.g(r_b["x"])[0], 0))
(r_e, dt_e) = timeit(lambda: sqp(**P.kwargs(), hess_L=P.hess_L, tol=1e-9)); fs["SQP-exact"] = (len(r_e["log"]), dt_e, P.f(r_e["x"]), max(P.g(r_e["x"])[0], 0))
for m in ("SLSQP", "trust-constr"):
    s, dt = timeit(lambda: scipy_run(m)); fs[m] = (s.nit, dt, P.f(s.x), max(P.g(s.x)[0], 0))
xg, dtg = timeit(lambda: penalty_gd(), n=1); fs["GD + pénalité (20000 it.)"] = (20000, dtg, P.f(xg), max(P.g(xg)[0], 0))
fstar = P.f(r_e["x"])
print(f"  {'méthode':28s} {'iters':>6s} {'temps(s)':>9s} {'f-f*':>10s} {'viol':>9s}")
for k, (it, dt, f, v) in fs.items():
    print(f"  {k:28s} {it:6d} {dt:9.4f} {f - fstar:10.1e} {v:9.1e}")

# ---------------------------------------------------------------- robustesse
print("\n=== Robustesse (moyenne ± écart-type) ===")
n_runs = 20
its, oks = [], []
for i in range(n_runs):                                  # points initiaux aléatoires
    x0 = rng.normal(scale=2.0, size=P.x0.size)
    r = sqp(**{**P.kwargs(), "x0": x0}, hess_L=P.hess_L)
    its.append(len(r["log"])); oks.append(r["converged"] and abs(P.f(r["x"]) - fstar) < 1e-7)
print(f"  {n_runs} points initiaux aléatoires (N(0,2^2)) : succès {sum(oks)}/{n_runs}, itérations {np.mean(its):.1f} ± {np.std(its):.1f}")

res = []
for p in (10, 30, 60, 100):                              # dimension croissante, features corrélées, bruit
    its_p, ts_p = [], []
    for seed in range(5):
        X, y = make_classification(n_samples=1000, n_features=p, n_informative=max(2, p // 3),
                                   n_redundant=p // 4, flip_y=0.1, class_sep=0.8, random_state=seed)
        X = StandardScaler().fit_transform(X)
        Pp = logistic_l2(X, y, 5.0)
        t0 = time.perf_counter(); r = sqp(**Pp.kwargs(), hess_L=Pp.hess_L); dt = time.perf_counter() - t0
        its_p.append(len(r["log"])); ts_p.append(dt)
    res.append((p, np.mean(its_p), np.std(its_p), np.mean(ts_p), np.std(ts_p)))
    print(f"  p={p:3d} (bruit 10%, features redondantes) : itérations {np.mean(its_p):.1f} ± {np.std(its_p):.1f}, "
          f"temps {np.mean(ts_p):.3f} ± {np.std(ts_p):.3f} s")

print("\n  Sensibilité à la tolérance (BFGS, t*):")
for tol in (1e-4, 1e-6, 1e-8, 1e-10):
    r = sqp(**P.kwargs(), tol=tol); print(f"    tol={tol:.0e} -> {len(r['log']):3d} iterations")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ps = [x[0] for x in res]
ax[0].errorbar(ps, [x[1] for x in res], yerr=[x[2] for x in res], fmt="o-"); ax[0].set(xlabel="dimension p", title="itérations SQP")
ax[1].errorbar(ps, [x[3] for x in res], yerr=[x[4] for x in res], fmt="o-"); ax[1].set(xlabel="dimension p", title="temps (s)")
[a.grid(alpha=.3) for a in ax]; plt.tight_layout(); plt.savefig("figures/robustness.png", dpi=150)
print("\nFigures : figures/path.png, figures/robustness.png")