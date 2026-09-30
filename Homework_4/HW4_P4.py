import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(".")
SEED = 20260930
N_SIM = 10_000

states = ["U", "I", "M", "F", "A"]
idx = {s: i for i, s in enumerate(states)}

P = np.array([
    [0.0, 1/2, 1/2, 0.0, 0.0],
    [1/4, 0.0, 0.0, 1/2, 1/4],
    [3/4, 0.0, 0.0, 0.0, 1/4],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])

h_exact = {"U": 1/2, "I": 5/8, "M": 3/8}
g_exact = {"U": 4.0, "I": 2.0, "M": 4.0}
tauF_exact = {"U": 4.0, "I": 9/5, "M": 5.0}
tauA_exact = {"U": 4.0, "I": 7/3, "M": 17/5}

Q = P[:3, :3]
R = P[:3, 3:5]

def discrete_inverse_transform(probabilities, rng):
    """Finite discrete inverse-transform sampler."""
    u = rng.random()
    cdf = np.cumsum(probabilities)
    j = int(np.searchsorted(cdf, u, side="right"))
    return min(j, len(probabilities) - 1)

def simulate_one(start_state, rng):
    s = idx[start_state]
    t = 0
    while s not in (idx["F"], idx["A"]):
        s = discrete_inverse_transform(P[s], rng)
        t += 1
    return states[s], t

def exact_conditional_pmf_from_I(n_values, fate):
    fate_col = 0 if fate == "F" else 1
    denom = h_exact["I"] if fate == "F" else 1 - h_exact["I"]
    eI = np.array([0.0, 1.0, 0.0])
    pmf = []
    for n in n_values:
        mass = eI @ np.linalg.matrix_power(Q, n - 1) @ R[:, fate_col]
        pmf.append(mass / denom)
    return np.asarray(pmf)

rng = np.random.default_rng(SEED)
records = []

for start in ["U", "I", "M"]:
    for rep in range(N_SIM):
        fate, T = simulate_one(start, rng)
        records.append((start, fate, T))

sim = pd.DataFrame(records, columns=["start", "fate", "T"])

rows = []
for start in ["U", "I", "M"]:
    d = sim[sim["start"] == start]
    folded = d[d["fate"] == "F"]
    aggregated = d[d["fate"] == "A"]

    rows.append({
        "start": start,
        "h_exact": h_exact[start],
        "h_hat": len(folded) / len(d),
        "g_exact": g_exact[start],
        "g_hat": d["T"].mean(),
        "tauF_exact": tauF_exact[start],
        "tauF_hat": folded["T"].mean(),
        "tauA_exact": tauA_exact[start],
        "tauA_hat": aggregated["T"].mean(),
        "n_folded": len(folded),
        "n_aggregated": len(aggregated),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(OUT / "folding_simulation_summary.csv", index=False)

I_runs = sim[sim["start"] == "I"]

for fate, long_name in [("F", "Folded"), ("A", "Aggregated")]:
    vals = I_runs.loc[I_runs["fate"] == fate, "T"].to_numpy()
    max_t = int(vals.max())
    n_plot = np.arange(1, max_t + 1)
    pmf = exact_conditional_pmf_from_I(n_plot, fate)

    plt.figure(figsize=(8, 5))
    bins = np.arange(0.5, max_t + 1.5, 1.0)
    plt.hist(
        vals,
        bins=bins,
        density=True,
        edgecolor="black",
        linewidth=0.8,
        label=f"Simulation ({len(vals)} runs)",
    )
    plt.plot(
        n_plot,
        pmf,
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Exact conditional PMF",
    )
    plt.xlabel("Absorption time T (steps)")
    plt.ylabel("Conditional probability mass")
    plt.title(f"Start I: T conditioned on {long_name.lower()}")
    plt.xticks(np.arange(1, max_t + 1, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"start_I_T_conditioned_{fate}.png", dpi=180)
    plt.close()
