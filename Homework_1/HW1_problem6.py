import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; comment out if running interactively
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(12345)
 
# ---------------------------------------------------------------
# 1. Analytic value from part (a)
# ---------------------------------------------------------------
ANALYTIC_P = 23 * np.pi / 192
print(f"Analytic probability (part a): {ANALYTIC_P:.6f}")
 
# ---------------------------------------------------------------
# 2. Monte Carlo estimate at a single N (vectorized)
# ---------------------------------------------------------------
def mc_estimate(N, rng):
    X = rng.random(N)
    Y = rng.random(N)
    Z = rng.random(N)
    hits = (X**2 + Y**2 < Z) & (Z**2 > X * Y)
    p_hat = hits.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N)  # normal-approx standard error
    return p_hat, se
 
# ---------------------------------------------------------------
# 3. Sweep sample size N (log-spaced) and record estimates
# ---------------------------------------------------------------
Ns = np.unique(np.logspace(1, 7, num=50, dtype=int))  # N = 1e2 ... 1e7
 
p_hats = np.empty(len(Ns))
ses = np.empty(len(Ns))
for i, N in enumerate(Ns):
    p_hats[i], ses[i] = mc_estimate(N, rng)
 
# ---------------------------------------------------------------
# 4. Plot: MC estimate vs N (log x-axis) with analytic value as
#    a horizontal reference line
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
 
ax.errorbar(Ns, p_hats, yerr=1.96 * ses, fmt="o-", ms=4, lw=1,
            capsize=2, label="Monte Carlo estimate (±95% CI)")
ax.axhline(ANALYTIC_P, color="black", ls="--", lw=1.5,
           label=f"Analytic value = 23π/192 ≈ {ANALYTIC_P:.5f}")
 
ax.set_xscale("log")
ax.set_xlabel("Sample size N")
ax.set_ylabel(r"Estimated $P(X^2+Y^2<Z,\ Z^2>XY)$")
ax.set_title("Monte Carlo convergence vs. analytic probability")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
 
plt.tight_layout()
plt.savefig("coupled_inequalities_mc.png", dpi=150)
print("Saved plot to coupled_inequalities_mc.png")
 
# ---------------------------------------------------------------
# 5. Print a summary table
# ---------------------------------------------------------------
print(f"\n{'N':>10} {'p_hat':>10} {'SE':>10} {'|p_hat-P|':>12}")
for N, p, se in zip(Ns[::4], p_hats[::4], ses[::4]):
    print(f"{N:>10d} {p:>10.5f} {se:>10.5f} {abs(p-ANALYTIC_P):>12.5f}")