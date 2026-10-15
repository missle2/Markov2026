"""HW6 #3(c),(d): will one case start an outbreak?  SIS-type epidemic chain
p_n = beta n (N-n) dt,  q_n = gamma n dt,  R0 = beta N / gamma,  T = hit {0, K}."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def h1(R0, N, K):
    """Exact P_1(X_T = K) = 1 / sum_{j=0}^{K-1} rho_j, rho_j = prod_{k<=j} N/(R0 (N-k))."""
    k = np.arange(1, K)
    logrho = np.concatenate(([0.0], np.cumsum(np.log(N/(R0*(N - k))))))
    a = logrho.max()                                   # log-sum-exp for safety
    return np.exp(-(a + np.log(np.exp(logrho - a).sum())))

# ---------------- (c) ----------------
R0s = np.linspace(0.5, 3, 501)
fig, ax = plt.subplots(figsize=(7, 4.5))
for N in (100, 1000):
    ax.plot(R0s, [h1(R, N, N//10) for R in R0s], label=f'exact $h_1$, N={N}, K={N//10}')
ax.plot(R0s, np.maximum(0, 1 - 1/R0s), 'k--', label=r'$\max(0,\,1-1/R_0)$')
ax.axvline(1, color='gray', lw=0.6)
ax.set_xlabel('$R_0$'); ax.set_ylabel(r'$h_1 = P_1(X_T = K)$')
ax.set_title('Probability one case reaches $K = N/10$ infected')
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig('outbreak_h1.png', dpi=150)
for N in (100, 1000):
    print(f'N={N:4d}: ' + '  '.join(f'h1(R0={R})={h1(R, N, N//10):.4f}' for R in (0.5, 1, 1.5, 2, 3)))

# ---------------- (d) ----------------
N, K, gamma, dt = 100, 10, 1.0, 0.005
n = np.arange(1, K)                                    # interior states 1..K-1

def mean_T(R0, dt=dt):
    """Solve -q_n e_{n-1} + (p_n+q_n) e_n - p_n e_{n+1} = 1, e_0 = e_K = 0."""
    beta = R0*gamma/N
    p, q = beta*n*(N - n)*dt, gamma*n*dt
    assert (p + q <= 1).all()
    ab = np.zeros((3, K - 1))
    ab[0, 1:] = -p[:-1]                                # superdiagonal  (e_{n+1})
    ab[1, :] = p + q                                   # diagonal
    ab[2, :-1] = -q[1:]                                # subdiagonal    (e_{n-1})
    return solve_banded((1, 1), ab, np.ones(K - 1))    # e_1..e_{K-1}, in steps

print('\n R0    e_1 (steps)   e_1*dt   e_1*dt with dt/10   h_1')
for R0 in (0.5, 0.8, 1.5, 3):
    e = mean_T(R0)
    print(f'{R0:4.1f} {e[0]:12.2f} {e[0]*dt:9.4f} {mean_T(R0, dt/10)[0]*dt/10:13.4f}'
          f' {h1(R0, N, K):11.4f}')

def simulate(R0, runs=200_000, seed=0):
    """Vectorized simulation of the discrete-time chain from X_0 = 1."""
    rng = np.random.default_rng(seed); beta = R0*gamma/N
    X = np.ones(runs, int); T = np.zeros(runs, int); alive = np.ones(runs, bool)
    while alive.any():
        x = X[alive]; u = rng.random(x.size)
        p, q = beta*x*(N - x)*dt, gamma*x*dt
        X[alive] = x + (u < p) - ((u >= p) & (u < p + q))
        T[alive] += 1
        alive &= (X > 0) & (X < K)
    return T*dt, X == K

R0 = 1.5
Tsim, hit = simulate(R0)
se = Tsim.std()/np.sqrt(Tsim.size)
print(f'\nSimulation, R0={R0}, {Tsim.size} runs: E[T]dt = {Tsim.mean():.4f} +/- {1.96*se:.4f} (95%),'
      f'  exact {mean_T(R0)[0]*dt:.4f};  P(hit K) = {hit.mean():.4f}, exact {h1(R0, N, K):.4f}')
