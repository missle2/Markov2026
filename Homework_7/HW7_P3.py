#!/usr/bin/env python3
"""Meteor shower: equivalent Poisson process constructions and numerical checks.

Dependencies: numpy, scipy, matplotlib.
Run: python meteor_simulation.py
All outputs are written to ./meteor_results/ alongside the script.

Rate is 4 meteors per hour. Random seeds are fixed for reproducibility.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma, binom
from scipy.special import gammaln

RATE = 4.0
N_TRIALS = 200_000
HORIZON = 1.0
SEED = 20261021
OUT = Path(__file__).resolve().parent / 'meteor_results'
OUT.mkdir(parents=True, exist_ok=True)


def simulate_by_gaps(n_trials: int, rate: float, rng):
    """Generate complete Poisson paths on [0,1], and T3 without truncating it.

    Vectorize the first three exponential gaps (including arrivals after t=1),
    then extend paths with a vectorized active-process loop until all next
    arrivals exceed the one-hour horizon. This retains the unconditional T3.
    """
    gaps = rng.exponential(scale=1.0 / rate, size=(n_trials, 3))
    initial_times = gaps.cumsum(axis=1)
    T3 = initial_times[:, 2].copy()
    first_time = initial_times[:, 0].copy()
    count = np.count_nonzero(initial_times <= HORIZON, axis=1).astype(np.int32)
    first_half = np.count_nonzero(initial_times <= HORIZON / 2, axis=1).astype(np.int32)
    raster = [sorted(initial_times[j, initial_times[j] <= HORIZON].tolist())
              for j in range(10)]

    clock = initial_times[:, 2].copy()
    active = clock <= HORIZON
    while np.any(active):
        ids = np.flatnonzero(active)
        clock[ids] += rng.exponential(scale=1.0 / rate, size=len(ids))
        arrived = clock[ids] <= HORIZON
        count[ids] += arrived
        first_half[ids] += (clock[ids] <= HORIZON / 2)
        for j in ids[(ids < 10) & arrived]:
            raster[int(j)].append(float(clock[j]))
        active[ids] = arrived

    return {'counts': count, 'first_half': first_half, 'T1': first_time,
            'T3': T3, 'raster': raster}


def simulate_by_count_uniform(n_trials: int, rate: float, rng):
    """Draw count N(1), then independently place that many uniform times."""
    count = rng.poisson(lam=rate * HORIZON, size=n_trials).astype(np.int32)
    ids = np.repeat(np.arange(n_trials, dtype=np.int32), count)
    locations = rng.uniform(0.0, HORIZON, size=len(ids))
    first_half = np.bincount(
        ids, weights=(locations <= HORIZON / 2).astype(np.int8),
        minlength=n_trials).astype(np.int32)
    first_time = np.full(n_trials, np.inf)
    np.minimum.at(first_time, ids, locations)
    ends = np.cumsum(count, dtype=np.int64)
    begins = ends - count
    raster = [sorted(locations[begins[j]:ends[j]].tolist()) for j in range(10)]
    return {'counts': count, 'first_half': first_half, 'T1': first_time,
            'raster': raster}


def diagnostic_summary(sim):
    counts = sim['counts']
    A = sim['first_half']
    B = counts - A
    sel = counts == 3
    return {
        'mean_N1': float(np.mean(counts)),
        'sample_variance_N1': float(np.var(counts, ddof=1)),
        'num_N1_equals_3': int(np.sum(sel)),
        'P_Nhalf_0_given_N1_3': float(np.mean(A[sel] == 0)),
        'mean_T1_given_N1_3_hours': float(np.mean(sim['T1'][sel])),
        'corr_first_and_second_half': float(np.corrcoef(A, B)[0, 1]),
        'mean_first_half': float(np.mean(A)),
        'mean_second_half': float(np.mean(B)),
    }


def total_variation(n, mean=4.0):
    """TV[Bin(n, mean/n), Poi(mean)] including Poisson tail above n."""
    k = np.arange(n + 1)
    p_bin = binom.pmf(k, n, mean / n)
    p_poi = poisson.pmf(k, mean)
    return float((np.sum(np.abs(p_bin - p_poi)) + poisson.sf(n, mean)) / 2)


def save_dict_csv(filename, records):
    if not records:
        return
    with (OUT / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'figure.dpi': 125, 'savefig.dpi': 190,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    rng_gap = np.random.default_rng(SEED)
    rng_uniform = np.random.default_rng(SEED + 1)

    gap = simulate_by_gaps(N_TRIALS, RATE, rng_gap)
    uniform = simulate_by_count_uniform(N_TRIALS, RATE, rng_uniform)
    s_gap = diagnostic_summary(gap)
    s_uniform = diagnostic_summary(uniform)
    p_t3_after_1 = float(poisson.cdf(2, RATE))

    exact = {'mean_N1': RATE, 'sample_variance_N1': RATE,
             'num_N1_equals_3': N_TRIALS * poisson.pmf(3, RATE),
             'P_Nhalf_0_given_N1_3': 1/8,
             'mean_T1_given_N1_3_hours': 1/4,
             'corr_first_and_second_half': 0.0,
             'mean_first_half': 2.0, 'mean_second_half': 2.0}
    records = [{'method': label, **data} for label, data in
               [('exponential_gaps', s_gap),
                ('poisson_count_uniform', s_uniform),
                ('theory', exact)]]
    save_dict_csv('simulation_summary.csv', records)

    # Figure 1: empirical distributions of counts versus the exact PMF.
    max_k = max(14, int(max(gap['counts'].max(), uniform['counts'].max())))
    k = np.arange(max_k + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.45), sharey=True, layout='constrained')
    for ax, (title, sim) in zip(axs,
            [('Exponential gaps', gap), ('Poisson count + uniforms', uniform)]):
        mass = np.bincount(sim['counts'], minlength=max_k + 1) / N_TRIALS
        ax.bar(k, mass, width=.78, alpha=.72, label='Simulation')
        ax.plot(k, poisson.pmf(k, RATE), 'ko', ms=4.2,
                label='Poisson(4) PMF', zorder=5)
        ax.set(title=title, xlabel='Number of meteors in one hour')
        ax.set_xticks(np.arange(0, max_k + 1, 2))
        ax.set_xlim(-.6, max_k + .7)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper right', frameon=False)
    axs[0].set_ylabel('Probability')
    fig.suptitle('One-hour meteor counts: two exact constructions')
    fig.savefig(OUT / 'figure1_counts_vs_poisson.png')
    plt.close(fig)

    # Figure 2: ten full realization rasters for each method.
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True,
                            layout='constrained')
    for ax, (title, sim) in zip(axs,
            [('Exponential gaps', gap), ('Poisson count + uniform positions', uniform)]):
        for row, events in enumerate(sim['raster'], start=1):
            ax.hlines(row, 0, 1, linewidth=.45, alpha=.27)
            if events:
                ax.vlines(events, row-.32, row+.32, linewidth=1.65)
        ax.set(title=title, xlabel='Time within hour (hours)',
               xlim=(0, 1), ylim=(.3, 10.7), yticks=np.arange(1, 11))
        ax.grid(axis='x', alpha=.2)
    axs[0].set_ylabel('Realization number')
    fig.suptitle('Ten realizations of each Poisson-process construction')
    fig.savefig(OUT / 'figure2_arrival_rasters.png')
    plt.close(fig)

    # Figure 3: T3. Important: no truncation of paths at t=1 for T3.
    t3 = gap['T3']
    fig, ax = plt.subplots(figsize=(8.8, 4.8), layout='constrained')
    grid = np.linspace(0, 3.6, 500)
    ax.hist(t3, bins=np.linspace(0, 3.6, 80), density=True, alpha=.65,
            label=f'Simulation (N={N_TRIALS:,})')
    ax.plot(grid, gamma.pdf(grid, a=3, scale=1/RATE), 'k-', linewidth=2.0,
            label=r'Gamma(shape=3, rate=4)')
    ax.axvline(3/RATE, ls='--', linewidth=1.3, label='Theoretical mean = 0.75 h')
    ax.set(xlabel='Third arrival time $T_3$ (hours)', ylabel='Probability density',
           title='Distribution of the third meteor arrival time', xlim=(0, 3.6))
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'figure3_T3_gamma.png')
    plt.close(fig)

    # Figure 4: observed conditional first-half counts and exact Binomial pmf.
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.5), sharey=True, layout='constrained')
    support = np.arange(4)
    for ax, (title, sim) in zip(axs,
            [('Exponential gaps', gap), ('Poisson count + uniforms', uniform)]):
        sample = sim['first_half'][sim['counts'] == 3]
        prob = np.bincount(sample, minlength=4)[:4] / len(sample)
        ax.bar(support, prob, alpha=.7, width=.7, label='Simulation')
        ax.plot(support, binom.pmf(support, 3, .5), 'ko', markersize=6,
                label='Binomial(3, 1/2)')
        ax.set(title=f'{title} ({len(sample):,} selected hours)',
               xlabel='Meteors in first half-hour', xticks=support, ylim=(0, .5))
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=.2)
    axs[0].set_ylabel('Conditional probability, given $N(1)=3$')
    fig.suptitle('Conditional binomial distribution of first-half counts')
    fig.savefig(OUT / 'figure4_conditional_binomial.png')
    plt.close(fig)

    # Figure 5: total variation between time-grid binomial and Poisson.
    nvalues = np.array([10, 15, 20, 25, 40, 50, 80, 100, 200, 400, 500, 1000, 2000])
    dt = 1.0 / nvalues
    tv = np.array([total_variation(int(n), RATE) for n in nvalues])
    coeffs = np.polyfit(np.log(dt), np.log(tv), 1)
    slope, intercept = coeffs
    tail_mask = nvalues >= 100
    slope_small_dt, intercept_small_dt = np.polyfit(
        np.log(dt[tail_mask]), np.log(tv[tail_mask]), 1)
    save_dict_csv('total_variation.csv',
                  [{'delta_t': float(h), 'num_steps': int(n),
                    'binomial_success_probability': float(RATE*h),
                    'total_variation': float(v)}
                   for n, h, v in zip(nvalues, dt, tv)])
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    ax.loglog(dt, tv, 'o-', linewidth=1.8, markersize=5,
              label='Exact discrete total variation')
    order = np.argsort(dt)
    ax.loglog(dt[order], np.exp(intercept) * dt[order] ** slope, '--',
              linewidth=1.5, label=f'Log–log fit, slope = {slope:.4f}')
    ax.loglog(dt[order], tv[-1] * (dt[order] / dt[-1]), ':',
              linewidth=1.5, label='Reference slope = 1')
    ax.set(xlabel=r'Time step $\Delta t$ (hours)', ylabel='Total variation distance',
           title=r'Binomial$(1/\Delta t,4\Delta t)$ vs Poisson$(4)$')
    ax.legend(frameon=False)
    ax.grid(which='both', alpha=.22)
    fig.savefig(OUT / 'figure5_total_variation_convergence.png')
    plt.close(fig)

    # Print and persist textual numerical results.
    result_lines = [
        f'Meteor Poisson process: rate {RATE:g} per hour; {N_TRIALS:,} trials per construction',
        f'P(N(0.25)=0) = {np.exp(-1):.9f}',
        f'P(T3>1) = P(N(1)<=2) = {p_t3_after_1:.9f}',
        'P(N(0.5)=0 | N(1)=3) = 0.125',
        'E[T1 | N(1)=3] = 0.25 h = 15 min',
        'Theoretical E[N(1)] = Var[N(1)] = 4',
        'Theoretical corr(N(0.5), N(1)-N(0.5)) = 0',
        f'Theoretical E[T3] = {3/RATE:.6f} h, simulation mean T3 = {t3.mean():.6f} h',
        f'Theoretical P(T3>1) = {p_t3_after_1:.8f}, simulation P(T3>1) = {np.mean(t3 > 1):.8f}',
        '',
    ]
    for title, val in [('Exponential gaps', s_gap), ('Poisson count + uniforms', s_uniform)]:
        result_lines += [title]
        for key, value in val.items():
            result_lines.append(f'  {key}: {value:.9f}' if isinstance(value, float)
                                else f'  {key}: {value}')
        result_lines.append('')
    result_lines += [
        f'Log-log TV fit over all {len(dt)} dt values: slope = {slope:.7f}',
        f'Log-log TV fit dt <= 0.01: slope = {slope_small_dt:.7f}',
        f'TV at dt=0.1: {tv[0]:.8f}',
        f'TV at dt=0.0005: {tv[-1]:.10f}',
        '', 'Time-step values and TV distances:',
    ]
    result_lines += [f'  {h:.7g}\t{v:.10f}' for h, v in zip(dt, tv)]
    (OUT / 'numerical_results.txt').write_text('\n'.join(result_lines) + '\n')
    print('\n'.join(result_lines))
    print(f'\nFigures, results, and CSV files saved in: {OUT}')


if __name__ == '__main__':
    main()
