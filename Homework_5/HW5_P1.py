import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb

def P(a, b, N=4):
    p = np.zeros((N+1, N+1))
    for k in range(N+1):
        if k < N: p[k, k+1] = a*(N-k)/N
        if k > 0: p[k, k-1] = b*k/N
        p[k, k] = 1 - p[k].sum()
    return p

def evolve(p, q0, nmax):
    Q = [q0]
    for _ in range(nmax): Q.append(Q[-1] @ p)
    return np.array(Q)

def binom(theta, N=4):
    return np.array([comb(N, k)*theta**k*(1-theta)**(N-k) for k in range(N+1)])

np.set_printoptions(precision=10, suppress=True)
d0 = np.eye(5)[0]

# (b) a = b = 1
pb = P(1, 1)
Qb = evolve(pb, d0, 60)
print("p (a=b=1):\n", pb)
print("q50 =", Qb[50]); print("q51 =", Qb[51])
run = np.cumsum(Qb[:, 2]) / np.arange(1, 62)
print("running avg of q_n(2) at n=60:", run[60], " pi(2) =", 6/16)
print("eig (a=b=1):", np.sort(np.linalg.eigvals(pb).real)[::-1])

# (c) a = 0.3, b = 0.1
a, b = 0.3, 0.1
pc = P(a, b)
A = np.vstack([(pc.T - np.eye(5))[:-1], np.ones(5)])   # pi(p - I) = 0, sum pi = 1
pi = np.linalg.solve(A, np.r_[np.zeros(4), 1.0])
pi_th = binom(a/(a+b))
print("pi (solve)   =", pi); print("pi (binomial)=", pi_th, " max diff", abs(pi-pi_th).max())
Qc = evolve(pc, d0, 400)
err = np.abs(Qc - pi).max(axis=1)
nstar = int(np.argmax(err < 1e-6))
print("smallest n with max|q_n - pi| < 1e-6:", nstar, " err[n-1], err[n] =", err[nstar-1], err[nstar])
ev = np.sort(np.abs(np.linalg.eigvals(pc)))[::-1]
print("eig moduli (a=.3,b=.1):", ev, " SLEM =", ev[1])

# plots
n = np.arange(61)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(n, Qb[:, 2], "o-", ms=3, label=r"$q_n(2)$")
ax[0].plot(n, Qb[:, 4], "s-", ms=3, label=r"$q_n(4)$")
ax[0].plot(n, run, "k-", lw=2, label=r"$\frac{1}{n+1}\sum_{k\leq n} q_k(2)$")
ax[0].axhline(6/16, color="gray", ls="--", label=r"$\pi(2)=3/8$")
ax[0].axhline(1/16, color="gray", ls=":", label=r"$\pi(4)=1/16$")
ax[0].set(title=r"(b) $a=b=1$, $q_0=\delta_0$ (period 2)", xlabel="n", ylabel="probability")
ax[0].legend(fontsize=8)
for k in range(5):
    l, = ax[1].plot(n, Qc[:61, k], label=fr"$q_n({k})$")
    ax[1].axhline(pi_th[k], color=l.get_color(), ls="--", lw=0.8)
ax[1].set(title=r"(c) $a=0.3,\ b=0.1$, $q_0=\delta_0$ (dashed: $\pi$)", xlabel="n", ylabel="probability")
ax[1].legend(fontsize=8)
m = np.arange(len(err))
ax[2].semilogy(m, err, label=r"$\max_k|q_n(k)-\pi(k)|$")
ax[2].semilogy(m, err[40]*0.9**(m-40), "k--", lw=0.8, label=r"$\propto 0.9^n$")
ax[2].axhline(1e-6, color="r", ls=":"); ax[2].axvline(nstar, color="r", ls=":")
ax[2].set(title=fr"(c) convergence, first $n$ below $10^{{-6}}$: {nstar}", xlabel="n", xlim=(0, 250), ylim=(1e-14, 2))
ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig("channel_markov.png", dpi=150)
