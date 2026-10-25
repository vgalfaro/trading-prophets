import numpy as np

def generate_basic_instance(T: int, spread_scale: float = 0.02):

    # 1. Generar el precio medio (mid) usando Lognormal
    mid_t = np.random.lognormal(mean=4.0, sigma=0.3, size=T)

    # 2. Generar un half-spread relativo siempre positivo usando una distribución Exponencial
    # (como fracción del mid, en promedio spread_scale ~ 2%). Se acota para que s_t > 0.
    half_spread_t = np.minimum(np.random.exponential(scale=spread_scale, size=T), 0.99)

    # 3. b_t = ask (lo que paga el trader para comprar) y s_t = bid (lo que recibe al vender),
    # de modo que s_t < b_t: comprar y vender en el mismo instante siempre pierde el spread
    b_t = mid_t * (1 + half_spread_t)
    s_t = mid_t * (1 - half_spread_t)

    tuplas_ask_bid = [(round(b, 2), round(s, 2)) for b, s in zip(b_t, s_t)]

    # print("\nPrecios con spread asegurado (s_t < b_t):")
    # print(tuplas_ask_bid[:5])

    return tuplas_ask_bid


def generate_distribution(K: int, spread_scale: float = 0.02, rng=None):
    """Distribución discreta D con K tipos (b^(k), s^(k)) y probabilidades p^(k),
    generados con el mismo modelo mid/spread que generate_basic_instance (s^(k) < b^(k))."""
    rng = np.random.default_rng(rng)
    mid = rng.lognormal(mean=4.0, sigma=0.3, size=K)
    half_spread = np.minimum(rng.exponential(scale=spread_scale, size=K), 0.99)
    b = mid * (1 + half_spread)
    s = mid * (1 - half_spread)
    p = rng.dirichlet(np.ones(K))  # p >= 0 y suma 1
    return b, s, p


def sample_instance(b, s, p, T: int, rng=None):
    """Muestrea T requests i.i.d. de D. Devuelve b_t, s_t y el tipo k_t de cada request
    (los algoritmos de la Sección 3 deciden según k_t)."""
    rng = np.random.default_rng(rng)
    k_t = rng.choice(len(p), size=T, p=p)
    return np.asarray(b)[k_t], np.asarray(s)[k_t], k_t


def generate_symmetric_distribution(K: int, rng=None):
    """Distribución i.i.d. simétrica (b^(k) = s^(k)), el caso del Teorema 1.5."""
    rng = np.random.default_rng(rng)
    price = rng.lognormal(mean=4.0, sigma=0.3, size=K)
    return price, price.copy(), rng.dirichlet(np.ones(K))


# ---------------------------------------------------------------------------
# Requests no idénticas (Sección 2). Todas las distribuciones se representan
# con arreglos (T, K): la fila t es D_{t+1} = {(b[t,k], s[t,k]) w.p. p[t,k]}.
# ---------------------------------------------------------------------------

def as_time_varying(b, s, p, T: int):
    """Lleva una distribución i.i.d. (arreglos (K,)) a la forma (T, K)."""
    K = len(p)
    return tuple(np.broadcast_to(np.asarray(v, dtype=float), (T, K)).copy() for v in (b, s, p))


def generate_noniid_distribution(T: int, K: int, spread_scale: float = 0.02, cycles: float = 2.0, rng=None):
    """D_1, ..., D_T distintas: el nivel del log-precio oscila en el tiempo y en cada t hay
    K tipos con su propio mid, spread y probabilidad (s < b como en generate_distribution)."""
    rng = np.random.default_rng(rng)
    level = 4.0 + 0.3 * np.sin(2 * np.pi * cycles * np.arange(T) / T)
    mid = np.exp(level[:, None] + 0.3 * rng.standard_normal((T, K)))
    half_spread = np.minimum(rng.exponential(scale=spread_scale, size=(T, K)), 0.99)
    p = rng.dirichlet(np.ones(K), size=T)
    return mid * (1 + half_spread), mid * (1 - half_spread), p


def generate_bernoulli_distribution(T: int, p_range=(0.2, 0.8), spread_scale: float = 0.02, rng=None):
    """Supuesto Bernoulli de la Sección 2: con prob. p_t llega (b_t, s_t) y si no llega
    (+inf, 0), donde la única acción útil es skip."""
    rng = np.random.default_rng(rng)
    b1, s1, _ = generate_noniid_distribution(T, 1, spread_scale, rng=rng)
    p_t = rng.uniform(*p_range, size=T)
    b = np.column_stack([b1[:, 0], np.full(T, np.inf)])
    s = np.column_stack([s1[:, 0], np.zeros(T)])
    p = np.column_stack([p_t, 1 - p_t])
    return b, s, p


def sample_batch(b, s, p, N: int, rng=None):
    """Muestrea N instancias de una vez. b, s, p son (T, K) (usar as_time_varying para el
    caso i.i.d.). Devuelve b_t, s_t, k_t de forma (N, T)."""
    rng = np.random.default_rng(rng)
    b, s, p = (np.asarray(v, dtype=float) for v in (b, s, p))
    T = p.shape[0]
    cum = np.cumsum(p, axis=1)
    cum[:, -1] = 1.0
    U = rng.random((N, T))
    k = (U[:, :, None] > cum[None]).sum(axis=2)
    t = np.arange(T)[None, :]
    return b[t, k], s[t, k], k


# ---------------------------------------------------------------------------
# Instancias de las cotas inferiores (Sección 4). Cada una es un dict con la
# distribución i.i.d. (b, s, p) y los parámetros T, B, B0.
# ---------------------------------------------------------------------------

def lb_instance_thm11(B: int):
    """Teorema 1.1: B0 = 0, T = B. Ningún algoritmo online gana más que 0."""
    q = 0.1 / B
    return dict(b=np.array([1.0, np.inf]), s=np.array([0.0, 2.0]), p=np.array([1 - q, q]),
                T=B, B=B, B0=0)


def lb_instance_thm14b(B0: int):
    """Teorema 1.4(b): con b = +inf el problema es una prophet inequality de B0 unidades.
    El paper sólo cita [HKS07]; esta instancia concreta es nuestra, armada como la del
    Teorema 1.5(b): T = 2 B0, valor 2 w.p. 1/2 (su número fluctúa ~sqrt(B0) en torno a B0),
    valor 1 w.p. 1/sqrt(B0) y 0 en otro caso. El online debe decidir si vender a 1 sin saber
    si después habrá suficientes valores 2. Requiere B0 >= 4."""
    r = 1 / np.sqrt(B0)
    if r > 0.5:
        raise ValueError('lb_instance_thm14b requiere B0 >= 4')
    return dict(b=np.full(3, np.inf), s=np.array([2.0, 1.0, 0.0]), p=np.array([0.5, r, 0.5 - r]),
                T=2 * B0, B=B0, B0=B0)


def lb_instance_thm15b(B0: int):
    """Teorema 1.5(b): precios simétricos, B = 1.5 T = 3 B0. Requiere B >= 16 (B0 >= 6)."""
    T, B = 2 * B0, 3 * B0
    r = 1 / np.sqrt(B)
    if r > 0.25:
        raise ValueError('lb_instance_thm15b requiere B0 >= 6')
    price = np.array([1.0, 0.0, 0.5])
    return dict(b=price, s=price.copy(), p=np.array([0.75, 0.25 - r, r]), T=T, B=B, B0=B0)


if __name__ == "__main__":
    generate_basic_instance(T=10)
