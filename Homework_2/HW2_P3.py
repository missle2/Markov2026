import numpy as np
import matplotlib.pyplot as plt
import time


def rejection_gamma2(N, lam, seed=42):
    """
    Generate N samples from Gamma(2,1), with density
        f(x) = x * exp(-x), x >= 0

    using acceptance-rejection with proposal
        g(x) = lam * exp(-lam*x).
    """

    rng = np.random.default_rng(seed)

    accepted = []
    proposals = 0

    # Envelope constant
    c = 1 / (lam * (1 - lam) * np.e)

    start_time = time.perf_counter()

    while len(accepted) < N:

        # Sample proposal X ~ Exponential(lam) by inversion
        U1 = rng.random()
        X = -np.log(1 - U1) / lam

        # Generate uniform for acceptance test
        U2 = rng.random()

        # Acceptance probability = f(X)/(c*g(X))
        acceptance_prob = (
            np.e * (1 - lam) * X
            * np.exp(-(1 - lam) * X)
        )

        proposals += 1

        if U2 < acceptance_prob:
            accepted.append(X)

    elapsed_time = time.perf_counter() - start_time

    accepted = np.array(accepted)

    return accepted, proposals, elapsed_time, c


# Number of accepted samples
N = 10_000


# Run for lambda = 0.5 and lambda = 0.2
results = {}

for lam in [0.5, 0.2]:

    samples, proposals, elapsed_time, c = rejection_gamma2(N, lam)

    empirical_acceptance = N / proposals
    theoretical_acceptance = 1 / c
    time_per_sample = elapsed_time / N

    results[lam] = samples

    print(f"\nlambda = {lam}")
    print(f"c = {c:.6f}")
    print(f"Theoretical acceptance = {theoretical_acceptance:.6f}")
    print(f"Empirical acceptance = {empirical_acceptance:.6f}")
    print(f"Total proposals = {proposals}")
    print(f"Mean time per accepted sample = "
          f"{time_per_sample * 1e6:.3f} microseconds")


# x values for plotting the true density
x = np.linspace(0, 12, 500)
f = x * np.exp(-x)


# Histogram for lambda = 0.5
plt.figure(figsize=(7, 4.5))

plt.hist(
    results[0.5],
    bins=50,
    density=True,
    alpha=0.6,
    label="Accepted samples"
)

plt.plot(
    x,
    f,
    linewidth=2,
    label=r"$f(x)=xe^{-x}$"
)

plt.xlabel("x")
plt.ylabel("Density")
plt.title(r"$\lambda=0.5$ Accept/Reject Sampling")
plt.legend()
plt.show()


# Histogram for lambda = 0.2
plt.figure(figsize=(7, 4.5))

plt.hist(
    results[0.2],
    bins=50,
    density=True,
    alpha=0.6,
    label="Accepted samples"
)

plt.plot(
    x,
    f,
    linewidth=2,
    label=r"$f(x)=xe^{-x}$"
)

plt.xlabel("x")
plt.ylabel("Density")
plt.title(r"$\lambda=0.2$ Accept/Reject Sampling")
plt.legend()
plt.show()