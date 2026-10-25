"""Chequeos de la implementación contra los lemas del paper.
Correr con `python3 test_paper.py` (o con pytest si está instalado)."""
import itertools
import numpy as np

from instances import (generate_distribution, generate_noniid_distribution, generate_bernoulli_distribution,
                       as_time_varying, sample_batch, lb_instance_thm11)
from offline import offline_opt, offline_opt_batch, recover_path, BUY, SELL
from lp import solve_lp_iid, solve_lp_relax
from decomposition import uncross, decompose, check_lemma_2_7, TOL
from algorithms import OnlineOptimal, Algorithm3, Theorem13, Theorem12
from simulate import simulate, evaluate

N = 20000  # réplicas Monte Carlo


def _within(x, target, z=4.0):
    """target está dentro de z errores estándar de la media de x."""
    return abs(x.mean() - target) <= z * x.std(ddof=1) / np.sqrt(len(x)) + 1e-9


def test_offline_dp_vs_bruteforce():
    rng = np.random.default_rng(0)
    T, B, B0 = 6, 2, 1
    for _ in range(20):
        b, s = rng.uniform(0, 10, T), rng.uniform(0, 10, T)
        best = -np.inf
        for seq in itertools.product((0, 1, -1), repeat=T):
            inv, profit = B0, 0.0
            for t, a in enumerate(seq):
                if not 0 <= inv + a <= B:
                    break
                inv += a
                profit += s[t] if a == -1 else -b[t] if a == 1 else 0.0
            else:
                best = max(best, profit)
        opt, _, policy = offline_opt(b, s, B, B0)
        actions, _ = recover_path(policy, B0)
        assert np.isclose(opt, best)
        assert np.isclose(s[actions == SELL].sum() - b[actions == BUY].sum(), opt)
        assert np.isclose(offline_opt_batch(b[None], s[None], B, B0)[0], opt)


def test_lp_iid_upper_bound():  # Lema 3.1
    rng = np.random.default_rng(1)
    for T, B, B0 in [(30, 1, 1), (40, 5, 2)]:
        b, s, p = generate_distribution(5, rng=rng)
        lp = solve_lp_iid(b, s, p, T, B, B0)
        opt = offline_opt_batch(*sample_batch(*as_time_varying(b, s, p, T), 5000, rng)[:2], B, B0)
        assert opt.mean() <= lp.value + 1e-6


def test_lp_relax_upper_bound():  # Lema 2.1
    rng = np.random.default_rng(2)
    for b, s, p in [generate_noniid_distribution(30, 3, rng=rng), generate_bernoulli_distribution(30, rng=rng)]:
        lp = solve_lp_relax(b, s, p)
        opt = offline_opt_batch(*sample_batch(b, s, p, 5000, rng)[:2], 1, 1)
        assert opt.mean() <= lp.value + 1e-6


def test_lp_relax_below_lp_iid():
    # Promediar en t una solución de LP Relax da una solución factible de LP_IID
    rng = np.random.default_rng(3)
    b, s, p = generate_distribution(4, rng=rng)
    T = 25
    assert solve_lp_relax(*as_time_varying(b, s, p, T)).value <= solve_lp_iid(b, s, p, T, 1, 1).value + 1e-6


def _algorithm1_literal(X):
    """Algoritmo 1 tal como está en el paper, sobre el acoplamiento agregado por tiempo."""
    X = X.copy()
    while True:
        pos = [tuple(ij) for ij in np.argwhere(X > TOL)]
        nested = next(((i, j, i2, j2) for (i, j) in pos for (i2, j2) in pos if i < i2 < j2 < j), None)
        if nested is None:
            return X
        i, j, i2, j2 = nested
        d = min(X[i, j], X[i2, j2])
        X[i, j] -= d; X[i2, j2] -= d
        X[i, j2] += d; X[i2, j] += d


def test_uncross_equals_algorithm1():  # Claim 2.6
    rng = np.random.default_rng(4)
    for _ in range(5):
        b, s, p = generate_noniid_distribution(10, 2, rng=rng)
        lp = solve_lp_relax(b, s, p)
        I, J, _, _, x = lp.trades
        X0 = np.zeros((11, 11))
        np.add.at(X0, (I, J), x)
        assert np.allclose(uncross(lp.qb, lp.qs), _algorithm1_literal(X0), atol=1e-7)


def test_decomposition():  # Claim 2.6, Lema 2.7, Lema 2.5.(i)
    rng = np.random.default_rng(5)
    for _ in range(5):
        lp = solve_lp_relax(*generate_noniid_distribution(60, 3, rng=rng))
        dec = decompose(lp)
        pos = np.argwhere(dec.X > TOL)
        assert np.all(pos[:, 0] < pos[:, 1])
        assert not any(i < i2 < j2 < j for i, j in pos for i2, j2 in pos)
        max_mass, max_cross = check_lemma_2_7(dec)
        assert max_mass <= 2.5 + 1e-6 and max_cross <= 1
        assert np.isclose(sum(dec.values), lp.value)


def test_alg4_holds_half():  # Lema 3.5
    rng = np.random.default_rng(6)
    b, s, p = generate_distribution(5, rng=rng)
    T = 40
    lp = solve_lp_iid(b, s, p, T, 1, 1)
    bt, st, kt = sample_batch(*as_time_varying(b, s, p, T), N, rng)
    r = simulate(Theorem13(lp, p, T, branch='alg4'), bt, st, kt, 1, 1, rng, record=True)
    assert np.all(np.abs(r.inventory.mean(axis=0) - 0.5) < 4 * 0.5 / np.sqrt(N))
    y1 = lp.y / (T * lp.alpha_S)
    assert _within(r.profit, 0.5 * T * (((lp.y - y1) * s).sum() - (lp.z * b).sum()))


def test_theorem13_guarantee():  # Teorema 1.3
    rng = np.random.default_rng(7)
    for _ in range(3):
        b, s, p = generate_distribution(5, rng=rng)
        T = 40
        lp = solve_lp_iid(b, s, p, T, 1, 1)
        res = evaluate({'alg': Theorem13(lp, p, T)}, *as_time_varying(b, s, p, T), 1, 1, N, rng)
        x = res['alg']
        assert x.mean() + 4 * x.std(ddof=1) / np.sqrt(N) >= lp.value / 4


def test_theorem12_branches_and_guarantee():  # Lema 2.3, Corolario 2.4, Lema 2.5.(ii), Teorema 1.2
    rng = np.random.default_rng(8)
    b, s, p = generate_noniid_distribution(40, 3, rng=rng)
    lp = solve_lp_relax(b, s, p)
    dec = decompose(lp)
    bt, st, kt = sample_batch(b, s, p, N, rng)
    # Ramas 2 y 3: exactamente LP / (4m) y P(tener copia) = 1/2 al inicio de cada segmento
    for c in (1, 2):
        probs = np.eye(3)[c]
        r = simulate(Theorem12(dec, lp.pf, probs=probs), bt, st, kt, 1, 1, rng, record=True)
        assert _within(r.profit, dec.values[c] / 10)
        for l, _ in dec.segments[c]:
            assert abs(r.inventory[:, l - 1].mean() - 0.5) < 4 * 0.5 / np.sqrt(N)
    # Rama 1: prophet inequality, 1/2 de LP(x~(1))
    r = simulate(Theorem12(dec, lp.pf, probs=(1, 0, 0)), bt, st, kt, 1, 1, rng)
    assert _within(r.profit, dec.values[0] / 2)
    r = simulate(Theorem12(dec, lp.pf), bt, st, kt, 1, 1, rng)
    assert _within(r.profit, lp.value / 22)


def test_online_optimal():
    rng = np.random.default_rng(9)
    b, s, p = generate_distribution(4, rng=rng)
    T, B, B0 = 30, 3, 1
    bT, sT, pT = as_time_varying(b, s, p, T)
    oo = OnlineOptimal(bT, sT, pT, B, B0)
    lp = solve_lp_iid(b, s, p, T, B, B0)
    res = evaluate({'oo': oo, 'alg3': Algorithm3(lp, p, T, B0, cap=B, c=1)}, bT, sT, pT, B, B0, N, rng)
    assert _within(res['oo'], oo.value)
    assert oo.value >= res['alg3'].mean() - 4 * res['alg3'].std(ddof=1) / np.sqrt(N)
    assert np.all(res['OPT offline'] >= res['oo'] - 1e-9)


def test_theorem11_online_zero():  # Teorema 1.1
    d = lb_instance_thm11(16)
    oo = OnlineOptimal(*as_time_varying(d['b'], d['s'], d['p'], d['T']), d['B'], d['B0'])
    assert abs(oo.value) < 1e-12


if __name__ == '__main__':
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith('test_')]
    for name, f in tests:
        f()
        print(f'ok  {name}')
    print(f'{len(tests)} tests ok')
