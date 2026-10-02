from dataclasses import dataclass
import numpy as np
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
