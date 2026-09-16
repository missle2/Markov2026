"""
Capture of the lamb (Redner & Krapivsky, Am. J. Phys. 67, 1277, 1999).

Parts (b) and (c): lamb starts at 0, N lions start at d0 = 10. Each step every
animal hops +-1 with probability 1/2. The lamb is caught the first time it
shares a site with any lion. We estimate S_N(t) = P(lamb alive at time t).

Implementation note: we simulate the separations D^(i) = lion_i - lamb.
Each step D^(i) changes by (eta_i - xi) in {-2, 0, +2}, where xi is the lamb's
hop (shared by all lions) and eta_i is lion i's hop. D starts even and moves by
even amounts, so it cannot jump over 0: capture <=> some D^(i) hits 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

R = 20_000        # independent lambs
T = 10_000        # time steps
D0 = 10           # initial lamb-lion separation
FIT_LO, FIT_HI = 1e2, 1e4
SEED = 12345


def survival(n_lions, R=R, T=T, d0=D0, seed=SEED):
    """Return S(t) for t = 0..T, estimated from R independent lambs."""
    rng = np.random.default_rng(seed)
    D = np.full((R, n_lions), d0, dtype=np.int32)   # separations, alive lambs only
    S = np.empty(T + 1)
    S[0] = 1.0
    for t in range(1, T + 1):
        m = D.shape[0]
        if m == 0:
            S[t:] = 0.0
            break
        xi = 2 * rng.integers(0, 2, size=(m, 1), dtype=np.int8) - 1        # lamb hops
        eta = 2 * rng.integers(0, 2, size=(m, n_lions), dtype=np.int8) - 1  # lion hops
        D += eta - xi
        alive = np.all(D != 0, axis=1)
        D = D[alive]                     # drop captured lambs
        S[t] = D.shape[0] / R
    return S


def fit_exponent(S, lo=FIT_LO, hi=FIT_HI):
    """Least-squares slope of log S vs log t over lo <= t <= hi; returns beta = -slope."""
    t = np.arange(len(S))
    mask = (t >= lo) & (t <= hi) & (S > 0)
    slope, intercept = np.polyfit(np.log(t[mask]), np.log(S[mask]), 1)
    return -slope, intercept


if __name__ == "__main__":
    t = np.arange(T + 1)

    # ---- (b) one lion ----
    S1 = survival(1, seed=SEED)
    beta1, c1 = fit_exponent(S1)
    S1_theory = erf(D0 / (2 * np.sqrt(np.maximum(t, 1e-12))))

    # ---- (c) two lions, both at d0 ----
    S2 = survival(2, seed=SEED + 1)
    beta2, c2 = fit_exponent(S2)

    # ---- one figure for (b) and (c) ----
    tt = t[1:]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(tt, S1[1:], lw=1.5, label=r"$S_1(t)$ simulation")
    ax.loglog(tt, S1_theory[1:], "k--", lw=1.2,
              label=r"$\mathrm{erf}(d_0/2\sqrt{t})$ (continuum)")
    ax.loglog(tt, S2[1:], lw=1.5, label=r"$S_2(t)$ simulation")
    ax.loglog(tt, S1[1:] ** 2, ":", lw=2, label=r"$S_1(t)^2$")

    fw = (t >= FIT_LO) & (t <= FIT_HI)
    ax.loglog(t[fw], np.exp(c1) * t[fw] ** (-beta1), color="C0", alpha=0.4, lw=5)
    ax.loglog(t[fw], np.exp(c2) * t[fw] ** (-beta2), color="C1", alpha=0.4, lw=5)
    ax.axvspan(FIT_LO, FIT_HI, color="gray", alpha=0.07)

    ax.set_xlabel("t (steps)")
    ax.set_ylabel("survival probability")
    ax.set_ylim(1e-4, 1.5)
    ax.set_title(rf"Lamb survival, $d_0={D0}$, $R={R}$:  "
                 rf"$\beta_1={beta1:.3f}$, $\beta_2={beta2:.3f}$  "
                 rf"(fit ${int(FIT_LO)}\leq t\leq {int(FIT_HI)}$)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("lamb_lions.png", dpi=150)

    # ---- table for (c) ----
    print(f"beta1 = {beta1:.4f}   (exact 1/2)")
    print(f"beta2 = {beta2:.4f}   (exact 3/4)")
    print(f"\n{'t':>7} {'S2(t)':>10} {'S1(t)^2':>10} {'ratio':>8}")
    for tk in (100, 1000, 10000):
        print(f"{tk:>7} {S2[tk]:>10.5f} {S1[tk]**2:>10.5f} {S2[tk]/S1[tk]**2:>8.3f}")
