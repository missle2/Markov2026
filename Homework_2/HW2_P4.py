"""
Two-state closed dwell time: f(t) = a*lf*exp(-lf t) + (1-a)*ls*exp(-ls t)
Composition sampling (Bernoulli + inverse transform), N = 1e5.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a, lf, ls = 0.9, 1000.0, 10.0
N = 100_000
rng = np.random.default_rng(20260909)

f = lambda t: a*lf*np.exp(-lf*t) + (1-a)*ls*np.exp(-ls*t)
S = lambda t: a*np.exp(-lf*t) + (1-a)*np.exp(-ls*t)   # survival P(T>t)

# ---- theory ----
ET   = a/lf + (1-a)/ls
ET2  = 2*a/lf**2 + 2*(1-a)/ls**2
VarT = ET2 - ET**2
CV   = np.sqrt(VarT)/ET

# ---- step 1-3 composition ----
U1 = rng.random(N)
U2 = rng.random(N)
lam = np.where(U1 < a, lf, ls)      # Bernoulli component label - where implements ifs
T = -np.log(U2)/lam                 # inverse transform Exp(lam)

# ---- diagnostics ----
t50 = 0.050
emp_mean, emp_var = T.mean(), T.var(ddof=1)
emp_tail = (T > t50).mean()
se_mean = T.std(ddof=1)/np.sqrt(N)
se_tail = np.sqrt(emp_tail*(1-emp_tail)/N)

print(f"integral of f (quad-free check): {a + (1-a):.6f}")
print(f"E[T]      theory {ET*1e3:10.4f} ms   empirical {emp_mean*1e3:10.4f} ms  (+/- {1.96*se_mean*1e3:.4f})")
print(f"Var(T)    theory {VarT:12.8f} s^2  empirical {emp_var:12.8f} s^2")
print(f"sd(T)     theory {np.sqrt(VarT)*1e3:10.4f} ms  empirical {np.sqrt(emp_var)*1e3:10.4f} ms")
print(f"CV        theory {CV:10.4f}      empirical {np.sqrt(emp_var)/emp_mean:10.4f}")
print(f"P(T>50ms) theory {S(t50):10.6f}      empirical {emp_tail:10.6f}  (+/- {1.96*se_tail:.6f})")
print(f"fraction of draws from fast component: {(lam==lf).mean():.5f} (target {a})")

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for k, (tmax, dt, ttl) in enumerate([(0.5, 5e-4, "full range, 0.5 ms bins"),
                                     (0.01, 5e-5, "first 10 ms, 0.05 ms bins")]):
    bins = np.arange(0, tmax + dt, dt)
    ax[k].hist(T, bins=bins, density=True, color="0.75", edgecolor="none",
               label="composition sample, $N=10^5$")
    tg = np.linspace(1e-6, tmax, 4000)
    ax[k].plot(tg, f(tg), "r-", lw=1.4, label=r"$f(t)$")
    ax[k].plot(tg, a*lf*np.exp(-lf*tg), "b--", lw=1.0, label="fast component")
    ax[k].plot(tg, (1-a)*ls*np.exp(-ls*tg), "g--", lw=1.0, label="slow component")
    ax[k].set_yscale("log")
    ax[k].set_xlabel("dwell time $t$ (s)")
    ax[k].set_ylabel("density (s$^{-1}$)")
    ax[k].set_title(ttl)
    ax[k].set_ylim(1e-2, 2e3)
    ax[k].legend(fontsize=8, frameon=False)

fig.suptitle("Closed dwell times: mixture of two exponentials", y=1.00)
fig.tight_layout()
fig.savefig("./dwell_time_mixture.png", dpi=160, bbox_inches="tight")
print("\nsaved ./dwell_time_mixture.png")