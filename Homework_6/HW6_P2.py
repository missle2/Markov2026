"""HW6 #2(c): Keener rankings for 2026 Big 12 with three priors (w = 1)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')                  # write PNGs without a display
import matplotlib.pyplot as plt
d = pd.read_csv('big12.csv')
teams = sorted(set(d.home) | set(d.away)); m = len(teams); ix = {t: k for k, t in enumerate(teams)}
h = lambda x: 0.5 + 0.5*np.sign(x - 0.5)*np.sqrt(abs(2*x - 1))

def results(df):                       # list of (i, j, h_ij) for both directions
    out = []
    for g in df.itertuples():
        i, j, S, T = ix[g.home], ix[g.away], g.home_pts, g.away_pts
        x = (S + 1)/(S + T + 2); out += [(i, j, h(x)), (j, i, h(1 - x))]
    return out

def keener(games, Pi, w=1.0):
    H = np.zeros((m, m)); n = np.zeros(m)
    for i, j, v in games: H[i, j] += v; n[i] += 1
    A = (H + w*Pi)/(n + w*(m - 1))[:, None]; np.fill_diagonal(A, 0)
    lam = np.linalg.eigvals(A); lam = lam[np.argsort(-abs(lam))]
    r = np.ones(m)/m                               # power iteration r <- Ar / sum(Ar)
    for k in range(500):
        rn = A @ r; rn /= rn.sum()
        if abs(rn - r).max() < 1e-14: break
        r = rn
    return r, lam[0].real, abs(lam[1])/lam[0].real, k, A

def plot_r(R, fname, title, order=None):
    """Grouped bar chart of r (as % of total) for each prior."""
    df = pd.DataFrame(R, index=teams)*100
    df = df.loc[order] if order is not None else df.sort_values(df.columns[-1], ascending=False)
    x = np.arange(len(df)); bw = 0.8/df.shape[1]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, col in enumerate(df.columns):
        ax.bar(x + (k - (df.shape[1] - 1)/2)*bw, df[col], bw, label=f'{col} prior')
    ax.axhline(100/m, color='k', lw=0.8, ls='--', label='uniform 1/m')
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel(r'$r_i$ (% of total)'); ax.set_title(title); ax.legend(ncol=4, fontsize=9)
    ax.grid(axis='y', alpha=0.3); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    return list(df.index)

played = d[(d.season == 2026) & (d.completed == 1)]
# priors
flat = np.full((m, m), 0.5)
S = np.zeros((m, m)); N = np.zeros((m, m))
for i, j, v in results(d[d.season == 2025]): S[i, j] += v; N[i, j] += 1
last = np.where(N > 0, S/np.maximum(N, 1), 0.5)
poll = {'Texas Tech':254,'BYU':239,'Arizona':207,'Houston':207,'Utah':198,'TCU':166,
        'Arizona State':161,'Kansas State':158,'Oklahoma State':124,'Baylor':88,
        'Cincinnati':85,'Kansas':79,'West Virginia':76,'UCF':68,'Colorado':40,'Iowa State':26}
s = np.array([poll[t] for t in teams], float); own = s[:, None]/(s[:, None] + s[None, :])
for P in (flat, last, own): assert np.allclose(P + P.T, 1)

R = {}
for name, P in [('flat', flat), ('2025', last), ('poll', own)]:
    r, lam, ratio, its, A = keener(results(played), P)
    rs = A.sum(1)
    print(f'{name:5s} lambda*={lam:.4f}  |l2/l*|={ratio:.4f}  iters={its}  '
          f'rowsum range=[{rs.min():.4f},{rs.max():.4f}]  A^2>0: {(A@A > 0).all()}')
    R[name] = r
tab = pd.DataFrame(R, index=teams).sort_values('poll', ascending=False)
print((tab*100).round(2).to_string())

wk = d[(d.season == 2026) & d.date.between('2026-10-10', '2026-10-11')]
print('\nPredictions (P(home) = r_h/(r_h+r_a))')
for g in wk.itertuples():
    row = [f'{g.away:>15s} @ {g.home:<15s}']
    for name in R:
        rh, ra = R[name][ix[g.home]], R[name][ix[g.away]]
        row.append(f'{name}: {g.home if rh > ra else g.away:<14s}({max(rh, ra)/(rh + ra):.3f})')
    print('  '.join(row))
tab.to_csv('rankings.csv')
order = plot_r(R, 'keener_r_2026.png', r'Keener ratings, 2026 Big 12 through Oct 4 ($w=1$)')

# ---- after the weekend: score the predictions, then refit with week 6 included ----
import os
if os.path.exists('week6_results.csv'):
    res = pd.read_csv('week6_results.csv')            # home, home_pts, away, away_pts
    print('\nScoring (Brier on p = r_h/(r_h+r_a); 0.25 = coin flip)')
    for name in R:
        ok, br = 0, 0.0
        for g in res.itertuples():
            p = R[name][ix[g.home]]/(R[name][ix[g.home]] + R[name][ix[g.away]])
            y = float(g.home_pts > g.away_pts); ok += (p > .5) == y; br += (p - y)**2
        print(f'{name:5s} {ok}/{len(res)} correct, Brier {br/len(res):.3f}')
    d2 = d.set_index(['season', 'home', 'away']).sort_index()
    for g in res.itertuples():
        d2.loc[(2026, g.home, g.away), ['home_pts', 'away_pts', 'completed']] = [g.home_pts, g.away_pts, 1]
    d2 = d2.reset_index()[d.columns]; d2.to_csv('big12_after_week6.csv', index=False)
    played2 = d2[(d2.season == 2026) & (d2.completed == 1)]
    R2 = {}
    for name, P in [('flat', flat), ('2025', last), ('poll', own)]:
        r, lam, ratio, its, A = keener(results(played2), P)
        print(f'{name:5s} after wk6: lambda*={lam:.4f} |l2/l*|={ratio:.4f}'); R2[name] = r
    plot_r(R2, 'keener_r_2026_after_wk6.png',
           r'Keener ratings, 2026 Big 12 through Oct 10 ($w=1$)', order=order)
    print((pd.DataFrame(R2, index=teams).sort_values('poll', ascending=False)*100).round(2).to_string())
