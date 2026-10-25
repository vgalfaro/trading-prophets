from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

_env = None

def _get_env():
    # Un solo Env para no revalidar la licencia (ni imprimir el banner) en cada LP
    global _env
    if _env is None:
        _env = gp.Env(empty=True)
        _env.setParam('OutputFlag', 0)
        _env.start()
    return _env


@dataclass
class LPIIDSolution:
    value: float            # LP(y*, z*), cota superior de E[OPT offline] (Lema 3.1)
    y: np.ndarray           # y_k: prob. de vender cuando llega el tipo k
    z: np.ndarray           # z_k: prob. de comprar cuando llega el tipo k
    alpha_S: float          # ventas esperadas por paso, ec. (7)
    alpha_B: float          # compras esperadas por paso, ec. (7)
    Gamma: float            # ventas esperadas totales = T * alpha_S, ec. (9)
    final_inventory: float  # B_0 + T * (alpha_B - alpha_S); el paper asume 0, ec. (8)


def solve_lp_iid(b, s, p, T, B, B_0):
    """Resuelve LP_IID (Sección 3.1) para la distribución D = {(b^(k), s^(k)) w.p. p^(k)}."""
    b, s, p = (np.asarray(v, dtype=float) for v in (b, s, p))
    K = len(p)
    can_buy = np.isfinite(b)  # b = inf: el tipo k solo permite vender

    model = gp.Model('lp_iid', env=_get_env())
    y = model.addVars(K, lb=0.0, name='y')  # vender
    z = model.addVars(K, lb=0.0, ub=[p[k] if can_buy[k] else 0.0 for k in range(K)], name='z')  # comprar

    model.setObjective(
        T * gp.quicksum(s[k] * y[k] for k in range(K))
        - T * gp.quicksum(b[k] * z[k] for k in range(K) if can_buy[k]),
        GRB.MAXIMIZE)

    model.addConstrs((y[k] + z[k] <= p[k] for k in range(K)), name='prob')  # (5)

    # (6): inventario final esperado en [0, B]. Basta imponerlo en t = T: como B_0 + t*(...)
    # es lineal en t y B_0 ∈ [0, B], si se cumple en t = 0 y t = T se cumple para todo t.
    final = B_0 + T * gp.quicksum(z[k] - y[k] for k in range(K))
    model.addConstr(final >= 0, name='inv_lb')
    model.addConstr(final <= B, name='inv_ub')

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f'LP_IID no resuelto a optimalidad (status {model.Status})')

    y_val = np.array([y[k].X for k in range(K)])
    z_val = np.array([z[k].X for k in range(K)])
    alpha_S, alpha_B = y_val.sum(), z_val.sum()
    sol = LPIIDSolution(value=model.ObjVal, y=y_val, z=z_val,
                        alpha_S=alpha_S, alpha_B=alpha_B, Gamma=T * alpha_S,
                        final_inventory=B_0 + T * (alpha_B - alpha_S))
    model.dispose()
    return sol


@dataclass
class LPRelaxSolution:
    value: float            # LP(x*), cota superior de E[OPT offline] (Lema 2.1)
    qb: np.ndarray          # (T+1, K) q^b_{i,k}: prob. de comprar en la request i con tipo k
    qs: np.ndarray          # (T+1, K) q^s_{j,k}: prob. de vender en la request j con tipo k
    bf: np.ndarray          # (T+1, K) precios de compra, fila 0 = request ficticia D_0
    sf: np.ndarray          # (T+1, K) precios de venta
    pf: np.ndarray          # (T+1, K) probabilidades
    trades: tuple           # (i, j, k, k', x) con x_{i,j,k,k'} > 0


def lp_value(qb, qs, bf, sf):
    """Objetivo (4): sum_i q^s_i s_i - sum_i q^b_i b_i (los b = inf con q^b = 0 no cuentan)."""
    return float((qs * sf).sum() - np.where(qb > 0, qb * np.where(np.isfinite(bf), bf, 0.0), 0.0).sum())


def solve_lp_relax(b, s, p, B_0=1):
    """Resuelve LP Relax (Sección 2) con K tipos por request (Sección C), para B = 1.

    b, s, p son (T, K): la fila t-1 describe la request t. Se agrega la request ficticia 0,
    D_0 = (0, 0) w.p. B_0, que representa la copia inicial. Variables x_{i,j,k,k'} >= 0 para
    0 <= i < j <= T: comprar en i (tipo k) y vender esa copia en j (tipo k')."""
    b, s, p = (np.asarray(v, dtype=float) for v in (b, s, p))
    T, K = p.shape
    bf = np.vstack([np.r_[0.0, np.full(K - 1, np.inf)], b])
    sf = np.vstack([np.zeros(K), s])
    pf = np.vstack([np.r_[float(B_0), np.zeros(K - 1)], p])
    can_buy = (pf > 0) & np.isfinite(bf)
    can_sell = pf > 0
    can_sell[0] = False

    # Una variable por (i < j, k, k') con compra en (i, k) y venta en (j, k') posibles
    I, J = np.triu_indices(T + 1, 1)
    pair, KB, KS = np.nonzero(can_buy[I][:, :, None] & can_sell[J][:, None, :])
    I, J = I[pair], J[pair]
    n = len(I)
    cols = np.arange(n)
    ones = np.ones(n)

    # Qb @ x = q^b_{i,k} y Qs @ x = q^s_{j,k} (filas i*K + k)
    Qb = sp.csr_matrix((ones, (I * K + KB, cols)), shape=((T + 1) * K, n))
    Qs = sp.csr_matrix((ones, (J * K + KS, cols)), shape=((T + 1) * K, n))
    # Agregados por tiempo: (Ab - As) @ x = q^b_i - q^s_i
    E = sp.csr_matrix((ones, (I, cols)), shape=(T + 1, n)) - sp.csr_matrix((ones, (J, cols)), shape=(T + 1, n))

    model = gp.Model('lp_relax', env=_get_env())
    x = model.addMVar(n, lb=0.0, name='x')
    model.setObjective((sf[J, KS] - bf[I, KB]) @ x, GRB.MAXIMIZE)

    model.addConstr((Qb + Qs) @ x <= pf.ravel(), name='prob')  # (1)

    # (2): sum_{i<t} sum_{j>=t} x_ij = sum_{i<t} q^b_i - sum_{j<t} q^s_j = H_t <= 1. Se
    # escribe con H_t = H_{t-1} + q^b_{t-1} - q^s_{t-1} para no tener O(T^3) coeficientes.
    H = model.addMVar(T, lb=-GRB.INFINITY, ub=1.0, name='H')
    D = sp.eye(T, format='csr') - sp.eye(T, k=-1, format='csr')
    model.addConstr(D @ H - E[:T] @ x == 0, name='hold')

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f'LP Relax no resuelto a optimalidad (status {model.Status})')

    xv = x.X
    nz = xv > 1e-12
    sol = LPRelaxSolution(value=model.ObjVal,
                          qb=(Qb @ xv).reshape(T + 1, K), qs=(Qs @ xv).reshape(T + 1, K),
                          bf=bf, sf=sf, pf=pf,
                          trades=(I[nz], J[nz], KB[nz], KS[nz], xv[nz]))
    model.dispose()
    return sol


if __name__ == "__main__":
    from instances import generate_distribution, sample_instance
    from offline import offline_opt

    T, K, B, B_0, N = 30, 10, 1, 1, 2000
    rng = np.random.default_rng(0)

    b, s, p = generate_distribution(K, rng=rng)
    lp = solve_lp_iid(b, s, p, T, B, B_0)
    print(f"LP_IID = {lp.value:.2f}   alpha_S = {lp.alpha_S:.4f}   alpha_B = {lp.alpha_B:.4f}   "
          f"Gamma = {lp.Gamma:.2f}   inventario final = {lp.final_inventory:.2e}")

    # Chequeo del Lema 3.1: E[OPT offline] <= LP_IID
    opts = np.array([offline_opt(*sample_instance(b, s, p, T, rng)[:2], B, B_0)[0] for _ in range(N)])
    se = opts.std(ddof=1) / np.sqrt(N)
    print(f"E[OPT offline] = {opts.mean():.2f} ± {1.96 * se:.2f}   (N = {N})")
    print(f"E[OPT] / LP_IID = {opts.mean() / lp.value:.3f}")
