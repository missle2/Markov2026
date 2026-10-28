#!/usr/bin/env python3
"""Exponential exam-finishing race: exact subset dynamic programming and simulation.

Run: python race_simulation.py
Requires: numpy, matplotlib (and optionally scipy for numerical infinity-limit check).
Figures, CSV summaries, and a text report are saved beside this script.
"""
from pathlib import Path
import csv
import math
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RATES = np.array([3.0, 2.0, 1.5, 1.2, 1.0, 0.75])
N_EXAMS = 200_000
SEED = 20261021


def subset_dp(rates):
    """Return E[time until all finish] and probability each student is last.

    For nonempty S, let r(S)=sum_{i in S} rate_i.
    E[S]=1/r(S) + sum_{i in S} rate_i/r(S) E[S\\{i}].
    L[S,j]=sum_{i in S} rate_i/r(S) L[S\\{i},j],
    with L[{j},j]=1, and zero elsewhere. Empty set has E=0.
    """
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    E = np.zeros(1 << n)
    last = np.zeros((1 << n, n))
    total_rate = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        low_bit = mask & (-mask)
        i = low_bit.bit_length() - 1
        total_rate[mask] = total_rate[mask ^ low_bit] + rates[i]
        if mask == low_bit:
            E[mask] = 1.0 / rates[i]
            last[mask, i] = 1.0
        else:
            E[mask] = 1.0 / total_rate[mask]
            for j in range(n):
                if mask & (1 << j):
                    remain = mask ^ (1 << j)
                    weight = rates[j] / total_rate[mask]
                    E[mask] += weight * E[remain]
                    last[mask, :] += weight * last[remain, :]
        assert abs(last[mask].sum() - 1.0) < 1e-11
    return E, last


def expectation_inclusion_exclusion(rates):
    """E[max X_i] = sum_{nonempty S} (-1)^{|S|+1} / sum_{i in S} lambda_i."""
    rates = np.asarray(rates, dtype=float)
    return sum(((-1.0) ** (r + 1)) / np.sum(rates[list(S)])
               for r in range(1, len(rates) + 1)
               for S in combinations(range(len(rates)), r))


def max_cdf_pdf(t, rates):
    """Exact CDF and density of M=max(X_i). t can be a vector."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(rates, dtype=float)
    exponent = np.exp(-t[:, None] * r[None, :])
    factors = -np.expm1(-t[:, None] * r[None, :])  # 1-exp(-rate*t)
    cdf = factors.prod(axis=1)
    # Product-of-other-factors avoids 0/0 at t=0.
    pdf = np.zeros_like(t)
    for i in range(len(r)):
        other = np.prod(np.delete(factors, i, axis=1), axis=1)
        pdf += r[i] * exponent[:, i] * other
    return cdf, pdf


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                         'axes.labelsize': 11, 'figure.dpi': 120,
                         'savefig.dpi': 180})

    rates = RATES
    lam = rates.sum()
    E, L = subset_dp(rates)
    full = (1 << len(rates)) - 1
    exact_E = E[full]
    exact_last = L[full]
    alt_E = expectation_inclusion_exclusion(rates)
    assert np.isclose(exact_E, alt_E, atol=1e-12)

    rng = np.random.default_rng(SEED)
    samples = rng.exponential(scale=1 / rates, size=(N_EXAMS, len(rates)))
    first_times = samples.min(axis=1)
    first_who = samples.argmin(axis=1)
    M = samples.max(axis=1)
    last_who = samples.argmax(axis=1)
    sample_E = M.mean()
    sample_E_se = M.std(ddof=1) / np.sqrt(N_EXAMS)
    sim_last = np.bincount(last_who, minlength=len(rates)) / N_EXAMS
    first_probs = rates / lam
    sim_first = np.bincount(first_who, minlength=len(rates)) / N_EXAMS

    cond_mean = np.zeros(len(rates))
    cond_se = np.zeros(len(rates))
    cond_n = np.zeros(len(rates), dtype=int)
    for i in range(len(rates)):
        condition = first_times[first_who == i]
        cond_mean[i] = np.mean(condition)
        cond_se[i] = np.std(condition, ddof=1) / np.sqrt(len(condition))
        cond_n[i] = len(condition)

    write_csv(OUT / 'six_student_results.csv',
              ['student', 'rate_per_hour', 'p_first_exact', 'p_first_sim',
               'p_last_exact', 'p_last_sim', 'n_first', 'mean_first_time_conditional', 'se_first_time_conditional'],
              [[i+1, rates[i], first_probs[i], sim_first[i],
                exact_last[i], sim_last[i], cond_n[i], cond_mean[i], cond_se[i]]
               for i in range(len(rates))])

    # Fig. 1: histogram versus exact density
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    xhi = max(8.5, float(np.quantile(M, .9995)))
    grid = np.linspace(0, xhi, 1200)
    _, exact_pdf = max_cdf_pdf(grid, rates)
    ax.hist(M, bins=105, range=(0, xhi), density=True,
            alpha=0.53, label=f'Simulation ({N_EXAMS:,} exams)', edgecolor='none')
    ax.plot(grid, exact_pdf, lw=2.6, label='Exact density')
    ax.axvline(exact_E, lw=1.4, linestyle='--', color='#454545',
               label=f'Exact mean = {exact_E:.4f} h')
    ax.set(xlabel='Time until last student finishes, M (hours)',
           ylabel='Probability density',
           title='Six-student exam: distribution of the last finishing time', xlim=(0, xhi))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'figure1_histogram_exact_density.png')
    plt.close(fig)

    # Fig. 2: last probabilities
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    positions = np.arange(len(rates))
    width = 0.37
    ax.bar(positions - width/2, exact_last, width=width, label='Exact: subset recursion')
    ax.bar(positions + width/2, sim_last, width=width, label='Simulation', alpha=.65)
    ax.set_xticks(positions, [f'{i+1}\n($\\lambda$={rates[i]:g})' for i in range(len(rates))])
    ax.set(ylabel='Probability of finishing last', xlabel='Student (rate per hour)',
           title='Who finishes last? Exact probabilities versus simulation')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'figure2_last_student_probabilities.png')
    plt.close(fig)

    # Fig. 3: empirical check of independence between first time and winner
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.errorbar(np.arange(1, len(rates)+1), cond_mean, yerr=1.96*cond_se,
                fmt='o', capsize=4, label='Mean first-finish time, conditional on identity (95% MC CI)')
    ax.axhline(1/lam, linestyle='--', color='#454545',
               label=f'Theoretical unconditional mean = 1/Λ = {1/lam:.5f} h')
    ax.set(xlabel='Identity of first student', ylabel='Mean first-finish time (hours)',
           title='First finishing time is independent of who finishes first',
           xticks=np.arange(1, len(rates)+1))
    ax.legend(frameon=False, loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figure3_independence_check.png')
    plt.close(fig)

    # Fig. 4: growing k and uniformly bounded expectation
    larger_rates = np.arange(1, 13, dtype=float)
    Ek_all, _ = subset_dp(larger_rates)
    ks = np.arange(1, 13)
    Ek = np.array([Ek_all[(1 << int(k)) - 1] for k in ks])
    bound = 2 * np.log(2)
    assert np.max(Ek) < bound
    fig, ax = plt.subplots(figsize=(9.1, 5.4))
    ax.plot(ks, Ek, marker='o', lw=2.0, label=r'Exact $\mathbb{E}[M_k]$, rates $1,2,\ldots,k$')
    ax.axhline(bound, ls='--', color='#454545', lw=1.6,
               label=rf'Uniform upper bound $2\ln 2={bound:.4f}$ h')
    ax.set(xlabel='Number of students, k', ylabel=r'$\mathbb{E}[M_k]$ (hours)',
           title='More students, faster rates: the expected last finish remains bounded',
           xticks=ks, ylim=(0.94, 1.43))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'figure4_mean_vs_k.png')
    plt.close(fig)
    write_csv(OUT / 'rate_i_equals_i_vs_k.csv', ['k', 'exact_expected_max_hours'], zip(ks, Ek))

    # Calculate approximate limiting value using the product over many rates.
    # This is not needed for the proof, just a numerical illustration.
    try:
        from scipy.integrate import quad
        def limiting_survival(t):
            if t <= 1e-7:
                return 1.0
            js = np.arange(1, 501)
            log_cdf = np.log1p(-np.exp(-js * t)).sum()
            return -np.expm1(log_cdf)
        infinite_limit, integerr = quad(limiting_survival, 0, 40, epsabs=5e-10)
    except ImportError:
        infinite_limit = float('nan')
        integerr = float('nan')

    report = []
    report.append('Exponential race: six independent students')
    report.append(f'rates per hour: {rates.tolist()}')
    report.append(f'Total rate Lambda = {lam:.6f} / hour; E[first time] = {1/lam:.6f} hour')
    report.append(f'Exact E[M] by 64-subset DP: {exact_E:.9f} hours')
    report.append(f'Exact E[M] by inclusion-exclusion: {alt_E:.9f} hours')
    report.append(f'Simulation E[M] from {N_EXAMS:,} exams: {sample_E:.9f} hours')
    report.append(f'Simulation standard error: {sample_E_se:.9f} hours')
    report.append(f'Simulation 95% Monte Carlo CI: [{sample_E-1.96*sample_E_se:.9f}, {sample_E+1.96*sample_E_se:.9f}] hours')
    report.append('Student | rate | first exact | first sim | last exact | last sim | first n | conditional first-time mean | SE')
    for i in range(len(rates)):
        report.append(f'{i+1:7d} | {rates[i]:4.2f} | {first_probs[i]:11.7f} | {sim_first[i]:9.7f} | '
                      f'{exact_last[i]:10.7f} | {sim_last[i]:8.7f} | {cond_n[i]:7d} | '
                      f'{cond_mean[i]:27.7f} | {cond_se[i]:.7f}')
    report.append('k | exact E[M_k], rates lambda_i=i')
    for k, value in zip(ks, Ek):
        report.append(f'{k:2d} | {value:.9f}')
    report.append(f'2 ln 2 upper bound: {bound:.9f}')
    report.append(f'Approximate k -> infinity limit: {infinite_limit:.9f} (quadrature estimated error: {integerr:.2e})')
    (OUT / 'numerical_report.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    print('\nSaved output files in:', OUT)


if __name__ == '__main__':
    main()
