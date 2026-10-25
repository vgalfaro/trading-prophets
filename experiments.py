"""Experimentos para contrastar los resultados del paper. Uso:

    python3 experiments.py all            # todos
    python3 experiments.py unit_iid       # uno en particular
    python3 experiments.py all --quick    # versión chica para probar que todo corre

Cada experimento guarda results/<nombre>.csv y results/<nombre>.png."""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from instances import (generate_distribution, generate_symmetric_distribution, as_time_varying,
                       generate_noniid_distribution, generate_bernoulli_distribution,
                       lb_instance_thm11, lb_instance_thm14b, lb_instance_thm15b)
from lp import solve_lp_iid, solve_lp_relax
from decomposition import decompose
from algorithms import OnlineOptimal, Algorithm3, Theorem13, Theorem12
from simulate import evaluate, summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _rows(res, lp_value=None, **meta):
    """Pasa el resultado de evaluate a filas de un DataFrame largo."""
    table = summarize(res)
    opt = res['OPT offline'].mean()
    rows = []
    for name, r in table.iterrows():
        rows.append(dict(meta, algoritmo=name, E_alg=r['E[ganancia]'], E_opt=opt,
                         ratio_opt=r['E[ALG]/E[OPT]'], ic_ratio=r['IC95 ratio'], LP=lp_value,
                         ratio_lp=r['E[ganancia]'] / lp_value if lp_value else np.nan))
    return rows


def _save(df, fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f'{name}.csv'), index=False)
    fig.savefig(os.path.join(RESULTS, f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> results/{name}.csv, results/{name}.png')


def _plot_ratios(df, guarantee, label, title):
    """Un punto por instancia: E[ALG]/LP y E[ALG]/E[OPT] por algoritmo."""
    algs = [a for a in df.algoritmo.unique() if a != 'OPT offline']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, col, ylabel in ((axes[0], 'ratio_lp', 'E[ALG] / LP'), (axes[1], 'ratio_opt', 'E[ALG] / E[OPT]')):
        for i, a in enumerate(algs):
            y = df.loc[df.algoritmo == a, col]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=18, alpha=0.7)
        ax.axhline(guarantee, color='red', ls='--', lw=1, label=label)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(algs, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Teorema 1.3: i.i.d., B = B0 = 1, ratio >= 1/4 contra LP_IID
# ---------------------------------------------------------------------------
def exp_unit_iid(quick=False, seed=0):
    rng = np.random.default_rng(seed)
    M, T, K, N = (3, 30, 4, 2000) if quick else (20, 50, 5, 10000)
    rows = []
    for inst in range(M):
        b, s, p = generate_distribution(K, rng=rng)
        lp = solve_lp_iid(b, s, p, T, 1, 1)
        bT, sT, pT = as_time_varying(b, s, p, T)
        policies = {'online óptimo': OnlineOptimal(bT, sT, pT, 1, 1),
                    'Thm 1.3': Theorem13(lp, p, T),
                    'Alg 3 (redondeo indep.)': Algorithm3(lp, p, T, 1, 1)}
        res = evaluate(policies, bT, sT, pT, 1, 1, N, rng)
        rows += _rows(res, lp.value, instancia=inst, T=T, K=K)
    df = pd.DataFrame(rows)
    fig = _plot_ratios(df, 0.25, 'garantía 1/4 (Thm 1.3)', f'i.i.d., B = B0 = 1 (T={T}, K={K}, {M} distribuciones)')
    _save(df, fig, 'unit_iid')
    return df


# ---------------------------------------------------------------------------
# Teorema 1.2: no i.i.d., B = B0 = 1, ratio >= 1/22 contra LP Relax
# ---------------------------------------------------------------------------
def exp_noniid(quick=False, seed=0):
    rng = np.random.default_rng(seed)
    M, T, K, N = (3, 25, 3, 2000) if quick else (15, 40, 3, 10000)
    rows = []
    for inst in range(M):
        kind = 'bernoulli' if inst % 3 == 2 else 'K tipos'
        b, s, p = generate_bernoulli_distribution(T, rng=rng) if kind == 'bernoulli' else \
            generate_noniid_distribution(T, K, rng=rng)
        lp = solve_lp_relax(b, s, p)
        dec = decompose(lp)
        policies = {'online óptimo': OnlineOptimal(b, s, p, 1, 1),
                    'Thm 1.2 (m=2.5)': Theorem12(dec, lp.pf),
                    'Thm 1.2 (m adaptivo)': Theorem12(dec, lp.pf, m='adaptive')}
        res = evaluate(policies, b, s, p, 1, 1, N, rng)
        rows += _rows(res, lp.value, instancia=inst, tipo=kind, T=T, intervalos=len(dec.intervals))
    df = pd.DataFrame(rows)
    fig = _plot_ratios(df, 1 / 22, 'garantía 1/22 (Thm 1.2)', f'no i.i.d., B = B0 = 1 (T={T}, {M} instancias)')
    _save(df, fig, 'noniid')
    return df


# ---------------------------------------------------------------------------
# Teoremas 1.4(a) y 1.5(a): capacidad grande, 1 - ratio = O(log B / sqrt(B))
# ---------------------------------------------------------------------------
def _plot_rate(df, param, title, ref_label):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a in [a for a in df.algoritmo.unique() if a != 'OPT offline']:
        d = df[df.algoritmo == a].groupby(param).ratio_opt.mean()
        ax.loglog(d.index, np.maximum(1 - d.values, 1e-4), 'o-', label=a)
    x = np.array(sorted(df[param].unique()), dtype=float)
    x = x[x > 1]
    ax.loglog(x, 0.5 * np.log(x) / np.sqrt(x), 'k--', lw=1, label=ref_label)
    ax.loglog(x, 0.5 / np.sqrt(x), 'k:', lw=1, label=r'$\propto 1/\sqrt{\cdot}$')
    ax.set_xlabel(param)
    ax.set_ylabel('1 - E[ALG] / E[OPT]')
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    return fig


def exp_large_asym(quick=False, seed=0, cs=(13, 1, 0.25)):
    """Teorema 1.4(a): precios asimétricos, Algoritmo 3 con capacidad B0. B = B0, T = 20 B0."""
    rng = np.random.default_rng(seed)
    B0s, M, K, N = ([1, 4, 16], 2, 4, 300) if quick else ([1, 2, 4, 8, 16, 32, 64, 128, 256], 5, 5, 600)
    rows = []
    for inst in range(M):
        b, s, p = generate_distribution(K, rng=rng)
        for B0 in B0s:
            B, T = B0, 20 * B0
            lp = solve_lp_iid(b, s, p, T, B, B0)
            bT, sT, pT = as_time_varying(b, s, p, T)
            policies = {'online óptimo': OnlineOptimal(bT, sT, pT, B, B0)}
            policies.update({f'Alg 3 (c={c})': Algorithm3(lp, p, T, B0, cap=B0, c=c) for c in cs})
            res = evaluate(policies, bT, sT, pT, B, B0, N, rng)
            rows += _rows(res, lp.value, instancia=inst, B0=B0, B=B, T=T, Gamma=lp.Gamma)
    df = pd.DataFrame(rows)
    fig = _plot_rate(df, 'B0', 'Asimétrico: Algoritmo 3 con capacidad B0 (Thm 1.4a)', r'$\propto \log B_0/\sqrt{B_0}$')
    _save(df, fig, 'large_asym')
    return df


def exp_large_sym(quick=False, seed=0, cs=(13, 1, 0.25)):
    """Teorema 1.5(a): precios simétricos, Algoritmo 3 con capacidad B. B0 = B/2, T = 4 B."""
    rng = np.random.default_rng(seed)
    Bs, M, K, N = ([2, 8, 32], 2, 4, 300) if quick else ([2, 4, 8, 16, 32, 64, 128, 256], 5, 5, 600)
    rows = []
    for inst in range(M):
        b, s, p = generate_symmetric_distribution(K, rng=rng)
        for B in Bs:
            B0, T = B // 2, 4 * B
            lp = solve_lp_iid(b, s, p, T, B, B0)
            bT, sT, pT = as_time_varying(b, s, p, T)
            policies = {'online óptimo': OnlineOptimal(bT, sT, pT, B, B0)}
            policies.update({f'Alg 3 (c={c})': Algorithm3(lp, p, T, B0, cap=B, c=c, symmetric=True) for c in cs})
            res = evaluate(policies, bT, sT, pT, B, B0, N, rng)
            rows += _rows(res, lp.value, instancia=inst, B=B, B0=B0, T=T)
    df = pd.DataFrame(rows)
    fig = _plot_rate(df, 'B', 'Simétrico: Algoritmo 3 con capacidad B (Thm 1.5a)', r'$\propto \log B/\sqrt{B}$')
    _save(df, fig, 'large_sym')
    return df


# ---------------------------------------------------------------------------
# Cotas inferiores (Sección 4): el online óptimo exacto contra el profeta
# ---------------------------------------------------------------------------
def exp_lower_bounds(quick=False, seed=0):
    rng = np.random.default_rng(seed)
    if quick:
        B11, B14, B15, N = [2, 8, 32], [16, 64], [8, 32], 1000
    else:
        B11, B14, B15, N = [2, 4, 8, 16, 32, 64, 128], [16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256], 4000
    rows = []
    for thm, maker, params, key in (('1.1', lb_instance_thm11, B11, 'B'),
                                    ('1.4b', lb_instance_thm14b, B14, 'B0'),
                                    ('1.5b', lb_instance_thm15b, B15, 'B0')):
        for v in params:
            d = maker(v)
            bT, sT, pT = as_time_varying(d['b'], d['s'], d['p'], d['T'])
            online = OnlineOptimal(bT, sT, pT, d['B'], d['B0'])
            res = evaluate({}, bT, sT, pT, d['B'], d['B0'], N, rng)
            opt = res['OPT offline']
            rows.append(dict(teorema=thm, param=key, valor=v, T=d['T'], B=d['B'], B0=d['B0'],
                             E_opt=opt.mean(), ic_opt=1.96 * opt.std(ddof=1) / np.sqrt(N),
                             online_opt=online.value,
                             ratio=online.value / opt.mean() if opt.mean() > 0 else np.nan))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    d = df[df.teorema == '1.1']
    axes[0].plot(d.valor, d.E_opt, 'o-', label='E[OPT offline]')
    axes[0].plot(d.valor, d.online_opt, 's-', label='online óptimo (exacto)')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('B')
    axes[0].set_title('Thm 1.1 (B0 = 0): el online no gana nada')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for thm, label in (('1.4b', 'Thm 1.4b: $(1-r)\\sqrt{B_0}$'), ('1.5b', 'Thm 1.5b: $(1-r)\\sqrt{B}$')):
        d = df[df.teorema == thm]
        scale = np.sqrt(d.B0) if thm == '1.4b' else np.sqrt(d.B)
        axes[1].plot(d.valor, (1 - d.ratio) * scale, 'o-', label=label)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('B0')
    axes[1].set_ylabel('(1 - online óptimo / E[OPT]) * sqrt(capacidad)')
    axes[1].set_title('Si es ~constante, el gap es Θ(1/sqrt(capacidad))')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(df, fig, 'lower_bounds')
    return df


EXPERIMENTS = {'unit_iid': exp_unit_iid, 'noniid': exp_noniid, 'large_asym': exp_large_asym,
               'large_sym': exp_large_sym, 'lower_bounds': exp_lower_bounds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=list(EXPERIMENTS) + ['all'])
    parser.add_argument('--quick', action='store_true', help='tamaños chicos para probar')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    names = list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    pd.set_option('display.width', 200)
    for name in names:
        print(f'== {name}')
        t0 = time.time()
        df = EXPERIMENTS[name](quick=args.quick, seed=args.seed)
        if 'algoritmo' in df:
            print(df.groupby('algoritmo')[['ratio_opt', 'ratio_lp']].agg(['mean', 'min']).round(3))
        else:
            print(df.round(4).to_string(index=False))
        print(f'  ({time.time() - t0:.1f} s)')
