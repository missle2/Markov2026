import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ---------- (a) link matrix read off the graph (pages 1..8 -> index 0..7)
links = {1: [2, 3], 2: [3, 4], 3: [1, 5], 4: [1, 3, 5],
         5: [2, 6, 7], 6: [6], 7: [8], 8: [7]}
N = 8
p = np.zeros((N, N))
for i, out in links.items():
    for j in out: p[i-1, j-1] = 1/len(out)
print("p =\n", p)

q = np.full(N, 1/N); Q = [q]
for n in range(101): Q.append(Q[-1] @ p)
print("q100 =", Q[100]); print("q101 =", Q[101])
w, V = np.linalg.eig(p.T)
print("eigenvalues of p:", np.round(np.sort_complex(w), 4))

# ---------- (b) Google matrix
d = 0.85
G = d*p + (1-d)/N*np.ones((N, N))
q = np.full(N, 1/N); n = 0
while True:
    qn = q @ G; n += 1
    if np.abs(qn - q).sum() < 1e-10: q = qn; break
    q = qn
pi_power = q
A = np.vstack([(G.T - np.eye(N))[:-1], np.ones(N)])
pi = np.linalg.solve(A, np.r_[np.zeros(N-1), 1.0])
print(f"power iteration: {n} iterations; ||pi_power - pi_solve||_1 = {np.abs(pi_power-pi).sum():.2e}")
print("pi =", pi)
order = np.argsort(-pi)
print("ranking:", [(int(i)+1, round(float(pi[i]), 6)) for i in order])
print("in-link weights to 1 and 5:", p[:, 0], p[:, 4])
print("bound check d^n*||q0-pi|| at n:", d**n*np.abs(np.full(N,1/N)-pi).sum())

# ---------- (c) R surfers, vectorized discrete inverse transform
rng = np.random.default_rng(2026)
R, T = 100, 10**5
cdf = np.cumsum(G, axis=1); cdf[:, -1] = 1.0
x = np.zeros(R, dtype=np.int64)                      # all start at page 1
counts = np.zeros((R, N))
checkpoints = np.unique(np.round(np.logspace(0, 5, 61)).astype(int))
ck = set(checkpoints); err_rms = []; rows = np.arange(R)
for t in range(1, T+1):
    u = rng.random(R)
    x = (u[:, None] > cdf[x]).sum(axis=1)            # smallest j with u <= F(x, j)
    counts[rows, x] += 1
    if t in ck:
        pihat = counts / t
        err_rms.append(np.sqrt(np.mean(np.max(np.abs(pihat - pi), axis=1)**2)))
err_rms = np.array(err_rms)
pihat = counts / T
print("surfer 0 estimate:", pihat[0])
print("mean over surfers :", pihat.mean(0))

m = (checkpoints >= 1e2) & (checkpoints <= 1e5)
slope, icpt = np.polyfit(np.log10(checkpoints[m]), np.log10(err_rms[m]), 1)
print(f"fitted slope (1e2..1e5): {slope:.3f}")

# asymptotic (CLT) variance via fundamental matrix Z = (I - G + 1 pi)^{-1}
Z = np.linalg.inv(np.eye(N) - G + np.outer(np.ones(N), pi))
sig2 = np.array([2*pi[i]*Z[i, i] - pi[i] - pi[i]**2 for i in range(N)])
iid = pi*(1-pi)
print("iid var pi(1-pi):      ", iid)
print("Markov asymptotic var: ", sig2)
print("inflation sigma^2/iid: ", sig2/iid)
print(f"err at T=1e5: {err_rms[-1]:.3e};  sqrt(pi6(1-pi6)/T) = {np.sqrt(iid[5]/T):.3e};"
      f"  sqrt(sigma6^2/T) = {np.sqrt(sig2[5]/T):.3e}")
lam6 = d + (1-d)/N
print("self-loop prob at 6:", lam6, " mean sojourn:", 1/(1-lam6))
emp_sd = pihat.std(0)
print("empirical sd over surfers at T:", emp_sd)
print("predicted sqrt(sigma^2/T):     ", np.sqrt(sig2/T))

# ---------- figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
k = np.arange(1, N+1); wdt = 0.38
ax[0].bar(k - wdt/2, pi, wdt, label=r"$\pi$ (linear solve)")
ax[0].bar(k + wdt/2, pihat[0], wdt, label=r"$\hat\pi$, one surfer, $T=10^5$")
ax[0].set(xticks=k, xlabel="page", ylabel="probability", title="PageRank, d = 0.85")
ax[0].legend()
ax[1].loglog(checkpoints, err_rms, "o", ms=4, label=r"rms$_R\ \max_i|\hat\pi_i-\pi_i|$")
TT = checkpoints[m]
ax[1].loglog(TT, 10**icpt*TT**slope, "k-", label=f"fit, slope {slope:.3f}")
ax[1].loglog(checkpoints, np.sqrt(iid[5]/checkpoints), "--", color="gray",
             label=r"$\sqrt{\pi_6(1-\pi_6)/T}$ (iid)")
ax[1].loglog(checkpoints, np.sqrt(sig2[5]/checkpoints), ":", color="r",
             label=r"$\sqrt{\sigma_6^2/T}$ (Markov CLT)")
ax[1].set(xlabel="T", ylabel="error", title=f"R = {R} surfers from page 1")
ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig("pagerank.png", dpi=150)
