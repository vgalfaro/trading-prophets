from dataclasses import dataclass
import numpy as np
from instances import sample_batch
from offline import offline_opt_batch, SKIP, BUY, SELL


@dataclass
class SimResult:
    profit: np.ndarray            # (N,)
    final_inventory: np.ndarray   # (N,)
    blocked: np.ndarray           # (N,) acciones no factibles (comprar lleno / vender vacío)
    actions: np.ndarray = None    # (N, T) si record=True
    inventory: np.ndarray = None  # (N, T+1) si record=True (inventory[:, t] = tras t requests)


def simulate(policy, b, s, k, B, B_0, rng=None, record=False):
    """Corre una política sobre N instancias ya muestreadas (b, s, k de forma (N, T))."""
    rng = np.random.default_rng(rng)
    N, T = b.shape
    inv = np.asarray(policy.reset(N, rng), dtype=int).copy()
    if np.any(inv > B_0) or np.any(inv < 0):
        raise ValueError('reset() debe conservar entre 0 y B_0 copias')
    profit = np.zeros(N)
    blocked = np.zeros(N, dtype=int)
    if record:
        actions = np.zeros((N, T), dtype=int)
        inventory = np.zeros((N, T + 1), dtype=int)
        inventory[:, 0] = inv
    for t in range(T):
        a = policy.act(t, k[:, t], b[:, t], s[:, t], inv, rng)
        buy = (a == BUY) & (inv < B) & np.isfinite(b[:, t])
        sell = (a == SELL) & (inv > 0)
        blocked += (a != SKIP) & ~buy & ~sell
        profit += np.where(sell, s[:, t], 0.0) - np.where(buy, b[:, t], 0.0)
        inv = inv + buy - sell
        if record:
            actions[:, t] = np.where(buy, BUY, np.where(sell, SELL, SKIP))
            inventory[:, t + 1] = inv
    if record:
        return SimResult(profit, inv, blocked, actions, inventory)
    return SimResult(profit, inv, blocked)


def evaluate(policies, b, s, p, B, B_0, N, rng=None, max_cells=2_000_000):
    """Monte Carlo con números aleatorios comunes: muestrea N instancias de D (b, s, p de forma
    (T, K)) y corre el profeta offline y cada política sobre las mismas instancias.
    Devuelve {nombre: ganancias (N,)}, incluyendo 'OPT offline'."""
    rng = np.random.default_rng(rng)
    T = np.asarray(p).shape[0]
    chunk = max(1, max_cells // T)  # para no tener matrices (N, T) gigantes en memoria
    out = {'OPT offline': []}
    out.update({name: [] for name in policies})
    for start in range(0, N, chunk):
        n = min(chunk, N - start)
        bt, st, kt = sample_batch(b, s, p, n, rng)
        out['OPT offline'].append(offline_opt_batch(bt, st, B, B_0))
        for name, policy in policies.items():
            out[name].append(simulate(policy, bt, st, kt, B, B_0, rng).profit)
    return {name: np.concatenate(v) for name, v in out.items()}


def summarize(profits, refs=None, base='OPT offline'):
    """Tabla con E[ganancia] ± IC 95% y el ratio E[ALG] / E[OPT] (con IC por método delta,
    usando que las instancias son comunes). refs: {nombre: valor} extra, p.ej. el LP."""
    import pandas as pd
    opt = profits[base]
    rows = []
    for name, x in profits.items():
        r = x.mean() / opt.mean()
        se_r = (x - r * opt).std(ddof=1) / (np.sqrt(len(x)) * abs(opt.mean()))
        row = {'algoritmo': name, 'E[ganancia]': x.mean(),
               'IC95': 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
               'E[ALG]/E[OPT]': r, 'IC95 ratio': 1.96 * se_r}
        for ref_name, v in (refs or {}).items():
            row[f'E[ALG]/{ref_name}'] = x.mean() / v
        rows.append(row)
    return pd.DataFrame(rows).set_index('algoritmo')
