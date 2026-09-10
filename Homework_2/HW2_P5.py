"""
Redirection-tree core caricature at mu = 1.
(b) inversion sampling of h(z) = 2(1-z)
(c) vectorized growth process, R realizations advanced in lockstep
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(5761)

h = lambda z: 2*(1-z)

# ---------- (a) theory ----------
mean_th, var_th = 1/3, 1/18
ratio_th = np.sqrt(var_th)/mean_th
print(f"(a) E[z]={mean_th:.6f}  Var={var_th:.6f}  sd={np.sqrt(var_th):.6f}  sd/mean={ratio_th:.6f}")

# ---------- (b) inversion ----------
M = 100_000
Z = 1 - np.sqrt(rng.random(M))          # H^{-1}(U) = 1 - sqrt(1-U), and 1-U =d U
print(f"(b) inversion sample: mean={Z.mean():.6f} (se {Z.std(ddof=1)/np.sqrt(M):.6f}), "
      f"sd/mean={Z.std(ddof=1)/Z.mean():.6f}")

# ---------- (c) growth process ----------
mu, N, R = 1.0, 10_000, 1_000
C = np.ones(R, dtype=np.int64)                       # C(3) = 1
for n in range(3, N):                                # single length-R draw per step
    C += (rng.random(R) < mu*C/n)
z = C/N

print(f"(c) N={N}, R={R}")
print(f"    mean z   = {z.mean():.6f}   (theory {mean_th:.6f}, se {z.std(ddof=1)/np.sqrt(R):.6f})")
print(f"    sd/mean  = {z.std(ddof=1)/z.mean():.6f}   (theory {ratio_th:.6f})")
print(f"    min core = {C.min()}  (z = {C.min()/N:.5f})")
print(f"    max core = {C.max()}  (z = {C.max()/N:.5f})")
print(f"    quartiles of C: {np.percentile(C,[25,50,75]).astype(int)}")

# ---------- plots ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
zg = np.linspace(0, 1, 400)

ax[0].hist(Z, bins=60, range=(0,1), density=True, color="0.78", edgecolor="none",
           label=r"inversion, $10^5$ draws")
ax[0].plot(zg, h(zg), "r-", lw=1.6, label=r"$h(z)=2(1-z)$")
ax[0].set_title("(b) inversion sampler")

ax[1].hist(z, bins=40, range=(0,1), density=True, color="0.78", edgecolor="none",
           label=r"$z=C/N$, $R=10^3$ runs")
ax[1].plot(zg, h(zg), "r-", lw=1.6, label=r"$h(z)=2(1-z)$")
ax[1].axvline(z.mean(), color="b", ls="--", lw=1.2, label=f"mean = {z.mean():.3f}")
ax[1].set_title(r"(c) growth process, $\mu=1$, $N=10^4$")

for a_ in ax:
    a_.set_xlabel("$z = C/N$"); a_.set_ylabel("density")
    a_.legend(frameon=False, fontsize=9); a_.set_xlim(0, 1)

fig.tight_layout()
fig.savefig("./core_growth.png", dpi=160, bbox_inches="tight")
print("saved ./core_growth.png")