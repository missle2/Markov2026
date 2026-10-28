#!/usr/bin/env python3
"""Lighthouse MCMC assignment, parts (c) and (d).

Usage:
    python lighthouse_mcmc.py

Dependencies: numpy, scipy, matplotlib.
Figures and CSV/text summaries are saved beside this script, in
lighthouse_results/.

Posterior convention:
    f(alpha, beta) is a density with respect to d(alpha) d(beta),
    g(alpha, b) = f(alpha, exp(b))*exp(b) is a density with respect
    to d(alpha) d(b), b=log(beta).
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

DATA = np.array([
    -1.77, 2.47, 1.34, 2.18, 13.05,
    1.60, 1.50, -1.97, 0.79, 1.50,
    2.00, 2.63, 1.19, -2.31, 0.87,
], dtype=float)
N = DATA.size
STEPS = 50_000
BURN = 5_000
STEP_SIZES = (0.02, 0.5, 5.0)
REPLICATES = 2_000
OUTPUT = Path(__file__).resolve().parent / "lighthouse_results"
OUTPUT.mkdir(exist_ok=True, parents=True)

# Visual styling
NAVY = "#25445f"
BLUE = "#327da8"
ORANGE = "#c26b40"
GREEN = "#3a8074"
GRAY = "#687786"
COLORS = (BLUE, GREEN, ORANGE)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "savefig.dpi": 200,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": .2,
})


def log_g(alpha: float, b: float) -> float:
    """Correct transformed log posterior, up to additive constant.

    Original prior: p(alpha, beta) proportional to 1/beta.
    Jacobian cancels the 1/beta prior, so log_g equals log likelihood
    as a function of alpha and beta=exp(b), ignoring constants.
    """
    beta_sq = np.exp(2.0 * b)
    return float(N * b - np.log(beta_sq + (DATA - alpha)**2).sum())


def run_metropolis(s: float, steps: int, rng: np.random.Generator):
    """Symmetric joint Gaussian random-walk Metropolis in (alpha, b)."""
    chain = np.empty((steps + 1, 2), dtype=float)
    z = np.array([0.0, 0.0])
    lp = log_g(*z)
    chain[0] = z
    accepted = 0

    for t in range(1, steps + 1):
        y = z + s * rng.standard_normal(2)
        lp_y = log_g(*y)
        # Work on log scale; avoids overflow in g(y)/g(z).
        if np.log(rng.random()) < lp_y - lp:
            z = y
            lp = lp_y
            accepted += 1
        chain[t] = z

    return chain, accepted / steps


def autocorrelation_fft(x):
    """Normalized empirical autocorrelation via FFT, O(T log T)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    m = 1 << (2*n - 1).bit_length()  # FFT padded to >= 2n
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conj(f), n=m)[:n].real
    # Divide by the number of overlapping pairs at each lag.
    acov /= np.arange(n, 0, -1)
    return acov / acov[0]


def integrated_autocorrelation_time(x, c=5.0):
    """Estimate tau_int = 1 + 2 sum_{lag>=1} rho(lag).

    Sokal self-consistent window: choose the first lag M satisfying
    M >= c*tau_int(M), where tau_int(M)=1+2*sum_{k=1}^M rho(k).
    If no valid window is found, this indicates a poorly estimated tau.
    """
    acf = autocorrelation_fft(x)
    max_lag = len(acf) // 2
    tau_windowed = 1.0 + 2.0 * np.cumsum(acf[1:max_lag+1])
    lag = np.arange(1, max_lag + 1)
    candidates = np.where((lag >= c * tau_windowed) & (tau_windowed > 0))[0]
    if len(candidates):
        idx = int(candidates[0])
        return float(tau_windowed[idx]), int(lag[idx]), True
    # Fallback when no consistent positive window is found.
    positives = np.where(tau_windowed > 0)[0]
    idx = int(positives[-1]) if len(positives) else 0
    return float(tau_windowed[idx]), int(lag[idx]), False


def quantiles_from_grid(x_grid, weights, qs=(.05, .95)):
    """Quantile interpolation of a discrete approximation to a density."""
    cum = np.cumsum(weights)
    cum = cum / cum[-1]
    return np.interp(qs, cum, x_grid)


def exact_grid():
    """Numerically normalize the *exact unnormalized formula* on a fine grid.

    This is quadrature, not a histogram of MCMC samples. Integrals
    are with respect to d(alpha) d(b), NOT d(alpha) d(beta).
    """
    alpha_grid = np.linspace(-8.0, 12.0, 1001)
    b_grid = np.linspace(-5.5, 3.6, 781)
    A = alpha_grid[None, :]
    B = b_grid[:, None]
    logpdf = N*B - sum(np.log(np.exp(2*B) + (A-xi)**2) for xi in DATA)

    da = alpha_grid[1] - alpha_grid[0]
    db = b_grid[1] - b_grid[0]
    logZ = logsumexp(logpdf) + np.log(da*db)
    pdf = np.exp(logpdf - logZ)
    masses = pdf * da * db  # probability of each grid cell
    marg_alpha = masses.sum(axis=0)
    marg_b = masses.sum(axis=1)

    alpha_mean = np.sum(alpha_grid * marg_alpha)
    beta_mean = np.sum(np.exp(b_grid) * marg_b)
    alpha_ci = quantiles_from_grid(alpha_grid, marg_alpha)
    beta_ci = np.exp(quantiles_from_grid(b_grid, marg_b))

    # Numerical sanity check: mass should not approach any boundary.
    boundary_mass = float(
        masses[:5,:].sum()+masses[-5:,:].sum()+
        masses[:, :5].sum()+masses[:, -5:].sum()
    )
    return dict(alpha_grid=alpha_grid, b_grid=b_grid, density=pdf,
                masses=masses, alpha_mean=alpha_mean, beta_mean=beta_mean,
                alpha_ci=alpha_ci, beta_ci=beta_ci,
                boundary_mass=boundary_mass)


def hpd_levels(grid, credible_probs=(0.5, 0.9, 0.99)):
    """Contour density thresholds enclosing specified posterior masses."""
    density = grid['density'].ravel()
    masses = grid['masses'].ravel()
    ids = np.argsort(density)[::-1]
    cumsum = np.cumsum(masses[ids])
    return {p: float(density[ids[np.searchsorted(cumsum, p)]])
            for p in credible_probs}


def plot_traces(chains, acceptance, taus):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, s, color in zip(axes, STEP_SIZES, COLORS):
        chain = chains[s]
        t = np.arange(400)
        ax.plot(t, chain[:400,0], color=color, linewidth=1.4)
        ax.axhline(np.median(DATA), color=GRAY, linestyle=':',
                   linewidth=1.1, label='Data median = 1.50 km')
        ax.text(.015, .93, f's = {s:g}   |   acceptance = {acceptance[s]:.1%}'
                f'   |   IAT(alpha) = {taus[s]:.0f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=.85,
                          edgecolor='none', pad=3))
        ax.set_ylabel(r'$\alpha$ (km)')
        ax.grid(alpha=.18)
    axes[-1].set_xlabel('MCMC step (first 400 saved states)')
    fig.suptitle('Lighthouse inference: impact of the joint proposal step size',
                 fontsize=15, weight='bold')
    fig.savefig(OUTPUT/'figure1_mcmc_traces.png', bbox_inches='tight')
    plt.close(fig)


def plot_posterior(samples, grid):
    ag, bg = grid['alpha_grid'], grid['b_grid']
    levels = hpd_levels(grid)
    fig, ax = plt.subplots(figsize=(9.4, 7), constrained_layout=True)
    ax.contourf(ag, bg, grid['density'], levels=24, cmap='Blues', alpha=.85)
    ordered_levels = [levels[.99], levels[.9], levels[.5]]
    cs = ax.contour(ag, bg, grid['density'], levels=ordered_levels,
                    colors=[NAVY, NAVY, NAVY], linewidths=[1.0, 1.5, 2.1])
    ax.clabel(cs, fmt={levels[.99]:'99% HPD',
                       levels[.9]:'90% HPD',
                       levels[.5]:'50% HPD'}, fontsize=9, inline=True)
    ax.scatter(samples[:,0], samples[:,1], s=2, c=ORANGE,
               alpha=.10, linewidths=0, rasterized=True,
               label=f'{len(samples):,} post-burn-in samples')
    ax.plot(grid['alpha_mean'],
            np.log(grid['beta_mean']), marker='*', markersize=14,
            markerfacecolor='white', markeredgecolor=NAVY,
            markeredgewidth=1.2, linestyle='none',
            label=r'$(E[\alpha],\log E[\beta])$ (grid)')
    ax.set_xlim(-2.7, 5.2)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel(r'$\alpha$ (km along shore)')
    ax.set_ylabel(r'$b=\log\beta$  (for $\beta$ in km)')
    ax.set_title('Exact posterior contours and joint Metropolis samples',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(alpha=.12)
    fig.savefig(OUTPUT/'figure2_joint_posterior.png', bbox_inches='tight')
    plt.close(fig)


def simulate_estimators(rng):
    """2,000 independent data sets for each of n=10, 100, 1000."""
    outcomes = {}
    for n in (10, 100, 1000):
        theta = rng.uniform(-np.pi/2, np.pi/2, size=(REPLICATES, n))
        x = 1.5 + 0.8*np.tan(theta)
        means = x.mean(axis=1)
        medians = np.median(x, axis=1)
        iqr_normalized = lambda values: (
            np.percentile(values, 75) - np.percentile(values, 25)) / 1.349
        outcomes[n] = dict(means=means, medians=medians,
                           mean_scatter=iqr_normalized(means),
                           median_scatter=iqr_normalized(medians))
    return outcomes


def plot_estimator_scatter(outcomes):
    ns = np.array(sorted(outcomes))
    mean_scatter = np.array([outcomes[n]['mean_scatter'] for n in ns])
    median_scatter = np.array([outcomes[n]['median_scatter'] for n in ns])
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    ax.plot(ns, mean_scatter, '-o', linewidth=2.3, markersize=8,
            color=ORANGE, label='Sample mean (simulated)')
    ax.plot(ns, median_scatter, '-o', linewidth=2.3, markersize=8,
            color=BLUE, label='Sample median (simulated)')
    ax.axhline(2 * 0.8 / 1.349, linestyle='--', linewidth=1.7,
               color=ORANGE, alpha=.75,
               label=r'Mean: exact Cauchy IQR / 1.349')
    ax.plot(ns, np.pi*0.8/(2*np.sqrt(ns)), ':', color=BLUE, lw=2,
            label=r'Median: asymptotic $\pi\beta/(2\sqrt{n})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel('Number of flashes per data set, n')
    ax.set_ylabel('Scatter: IQR / 1.349 (km)')
    ax.set_title('Cauchy sample mean does not improve with sample size',
                 fontsize=14, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(which='both', alpha=.2)
    fig.savefig(OUTPUT/'figure3_estimator_scatter.png', bbox_inches='tight')
    plt.close(fig)


def plot_estimator_boxes(outcomes):
    """Robust boxplots: fliers hidden because Cauchy tails can be huge."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True,
                             constrained_layout=True)
    for ax, key, color, title in [
        (axes[0], 'means', ORANGE, 'Sample mean'),
        (axes[1], 'medians', BLUE, 'Sample median'),
    ]:
        values = [outcomes[n][key] for n in (10, 100, 1000)]
        box = ax.boxplot(values, tick_labels=['10','100','1000'],
                         showfliers=False, patch_artist=True,
                         widths=.5, medianprops={'color':NAVY, 'linewidth':2})
        for patch in box['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(.22)
            patch.set_edgecolor(color)
        for whisker in box['whiskers']:
            whisker.set_color(color)
        for cap in box['caps']:
            cap.set_color(color)
        ax.axhline(1.5, color=NAVY, linestyle='--', alpha=.8,
                   label=r'True $\alpha=1.5$')
        ax.set_xlabel('n flashes')
        ax.set_title(title, fontsize=13)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper right', fontsize=9)
    axes[0].set_ylabel('Estimate of location (km)')
    fig.suptitle('2,000 independent Cauchy data sets per sample size\n'
                 '(boxplots show IQR; extreme outliers suppressed)',
                 fontsize=14, weight='bold')
    fig.savefig(OUTPUT/'figure4_estimator_boxplots.png', bbox_inches='tight')
    plt.close(fig)


def main():
    mcmc_rng = np.random.default_rng(20261021)
    simulation_rng = np.random.default_rng(20261022)
    chains, acceptance, taus, tau_windows, reliable = {}, {}, {}, {}, {}
    for s in STEP_SIZES:
        chain, ar = run_metropolis(s, STEPS, mcmc_rng)
        tau, cutoff, successful = integrated_autocorrelation_time(
            chain[BURN+1:, 0])
        chains[s], acceptance[s], taus[s] = chain, ar, tau
        tau_windows[s] = cutoff
        reliable[s] = successful and (STEPS - BURN) >= 50*tau

    grid = exact_grid()
    best = min(STEP_SIZES, key=lambda s: taus[s])
    selected = chains[best][BURN+1:]
    a = selected[:,0]
    beta = np.exp(selected[:,1])

    plot_traces(chains, acceptance, taus)
    plot_posterior(selected, grid)
    outcomes = simulate_estimators(simulation_rng)
    plot_estimator_scatter(outcomes)
    plot_estimator_boxes(outcomes)

    lines = []
    def report(t=''):
        lines.append(t)
        print(t)

    report('LIGHTHOUSE: REPRODUCIBLE NUMERICAL RESULTS')
    report(f'Observed flashes: n={N}, data mean={DATA.mean():.6f}, '
           f'data median={np.median(DATA):.6f}')
    report(f'Metropolis: start=(0,0), moves={STEPS}, burn-in={BURN}, '
           'IAT from remaining 45,000 alpha states')
    report('IAT convention: tau_int=1+2*sum_{k>=1} rho_alpha(k); '
           'Sokal self-consistent window c=5')
    report('s           acceptance    IAT_alpha    window_lag    >=50 tau?')
    for s in STEP_SIZES:
        report(f'{s:<11g} {acceptance[s]:>9.4%} '
               f'{taus[s]:>12.2f} {tau_windows[s]:>13} '
               f'{str(reliable[s]):>10}')
    report(f'Best proposal by lowest estimated IAT: s={best:g}')
    report(f'Posterior grid boundary mass (5-cell edge bands): '
           f'{grid["boundary_mass"]:.3e}')

    with (OUTPUT/'mcmc_diagnostics.csv').open('w', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['step_size','acceptance_rate','iat_alpha','window_lag',
                         'has_50_iat_lengths'])
        for s in STEP_SIZES:
            writer.writerow([s, acceptance[s], taus[s],
                             tau_windows[s], reliable[s]])

    report('')
    report('Posterior summary (equal-tailed 90% credible intervals, units km)')
    report('                       mean                 5%                 95%')
    for method, a_mean, b_mean, a_ci, b_ci in [
        ('MCMC', a.mean(), beta.mean(), np.quantile(a,[.05,.95]),
         np.quantile(beta,[.05,.95])),
        ('Grid', grid['alpha_mean'], grid['beta_mean'],
         grid['alpha_ci'], grid['beta_ci']),
    ]:
        report(f'{method:4s}   alpha  {a_mean:>16.5f}   '
               f'{a_ci[0]:>16.5f}   {a_ci[1]:>16.5f}')
        report(f'{method:4s}   beta   {b_mean:>16.5f}   '
               f'{b_ci[0]:>16.5f}   {b_ci[1]:>16.5f}')

    with (OUTPUT/'posterior_summary.csv').open('w', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['method','parameter','posterior_mean','credible_5pct',
                         'credible_95pct'])
        writer.writerow(['MCMC','alpha',a.mean(),*np.quantile(a,[.05,.95])])
        writer.writerow(['MCMC','beta',beta.mean(),*np.quantile(beta,[.05,.95])])
        writer.writerow(['Grid','alpha',grid['alpha_mean'],*grid['alpha_ci']])
        writer.writerow(['Grid','beta',grid['beta_mean'],*grid['beta_ci']])

    report('')
    report('2,000 simulated data sets per n; true (alpha, beta)=(1.5,0.8)')
    report('n          IQR(mean)/1.349      IQR(median)/1.349')
    with (OUTPUT/'estimator_scatter.csv').open('w', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['n','mean_iqr_over_1p349','median_iqr_over_1p349'])
        for n, values in outcomes.items():
            report(f'{n:<7d} {values["mean_scatter"]:>18.5f} '
                   f'{values["median_scatter"]:>22.5f}')
            writer.writerow([n, values['mean_scatter'],
                             values['median_scatter']])
    report(f'Theoretical scatter of sample mean for all n: '
           f'{2*0.8/1.349:.5f}')
    report('Asymptotic sample median SD: pi*beta/(2*sqrt(n))')
    (OUTPUT/'results.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    report('')
    report(f'All figures and tables saved to: {OUTPUT}')


if __name__ == '__main__':
    main()
