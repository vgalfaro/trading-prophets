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


if __name__ == "__main__":
    generate_basic_instance(T=10)
