"""Políticas online. Todas están vectorizadas sobre N réplicas y siguen la misma interfaz:

    reset(N, rng) -> copias iniciales que se conservan, forma (N,) (las demás se descartan;
                     los algoritmos del paper "botan" la copia inicial con prob. 1/2)
    act(t, k, b, s, inv, rng) -> acción de cada réplica en la request t (0-indexada), forma (N,)

k, b, s, inv son el tipo, los precios y el inventario actual de cada réplica. Las acciones
son SKIP / BUY / SELL de offline.py; el simulador ignora las que no son factibles."""
import math
import numpy as np
from offline import SKIP, BUY, SELL


def _ratio(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


class OnlineOptimal:
    """Política online óptima (no está en el paper; es la mejor cota para cualquier algoritmo
    online): Bellman sobre (t, inventario) promediando sobre el tipo de la request.
    b, s, p son (T, K). self.value es su ganancia esperada exacta."""

    def __init__(self, b, s, p, B, B_0):
        b, s, p = (np.asarray(v, dtype=float) for v in (b, s, p))
        T, K = p.shape
        self.B_0 = B_0
        self.action = np.zeros((T, B + 1, K), dtype=int)
        codes = np.array([SKIP, BUY, SELL])
        W = np.zeros(B + 1)  # W[x] = ganancia esperada óptima desde t con inventario x
        for t in range(T - 1, -1, -1):
            cand = np.full((3, B + 1, K), -np.inf)
            cand[0] = W[:, None]
            cand[1, :-1] = W[1:, None] - b[t][None, :]
            cand[2, 1:] = W[:-1, None] + s[t][None, :]
            self.action[t] = codes[cand.argmax(axis=0)]
            W = (cand.max(axis=0) * p[t][None, :]).sum(axis=1)
        self.value = W[B_0]

    def reset(self, N, rng):
        return np.full(N, self.B_0)

    def act(self, t, k, b, s, inv, rng):
        return self.action[t, inv, k]


class Algorithm3:
    """Algoritmo 3 (i.i.d., capacidad grande): redondeo independiente de (y*, z*) y, desde
    t > tau = T - ceil(c log B sqrt(T / alpha_S)), sólo vende para vaciar el inventario.

    cap: capacidad con la que corre el algoritmo. Teorema 1.4(a) (asimétrico): cap = B0.
         Teorema 1.5(a) (simétrico): cap = B y symmetric=True.
    c:   constante de tau; el paper usa 13.
    symmetric: vuelve tight la restricción (5) subiendo y_k, z_k por igual (Sección 3.3)."""

    def __init__(self, lp, p, T, B_0, cap, c=13.0, symmetric=False):
        p = np.asarray(p, dtype=float)
        y, z = lp.y.copy(), lp.z.copy()
        if symmetric:
            slack = np.maximum(p - y - z, 0.0)
            y, z = y + slack / 2, z + slack / 2
        self.alpha_S = y.sum()
        self.Gamma = T * self.alpha_S
        # "si B > 2 Gamma, B <- 2 Gamma" (demostración del Teorema 3.2)
        self.cap = int(max(1, min(cap, math.floor(2 * self.Gamma))))
        self.tau = max(0, T - math.ceil(c * math.log(self.cap) * math.sqrt(T / self.alpha_S)))
        self.B_0 = B_0
        self.buy_thr = _ratio(z, p)        # compra si rho <= z_k / p_k
        self.sell_thr = 1 - _ratio(y, p)   # vende si rho >= 1 - y_k / p_k

    def reset(self, N, rng):
        return np.full(N, self.B_0)

    def act(self, t, k, b, s, inv, rng):
        rho = rng.random(len(k))
        a = np.full(len(k), SKIP)
        a[(rho <= self.buy_thr[k]) & (self.buy_thr[k] > 0) & (t < self.tau) & (inv < self.cap)] = BUY
        a[(rho >= self.sell_thr[k]) & (self.sell_thr[k] < 1) & (inv > 0)] = SELL
        return a


class Theorem13:
    """Teorema 1.3 (i.i.d., B = B0 = 1, ratio 1/4). Separa (y*, z*) en
    (y', z') = (y* / (T alpha_S), 0)  -> Lema 3.4: vender la copia inicial (prophet inequality)
    (y'', z'') = (y* - y', z*)        -> Algoritmo 4: botar la copia con prob. 1/2 y redondear
    y corre cada uno con prob. 1/2. branch='pi' o 'alg4' fuerza una de las dos ramas."""

    def __init__(self, lp, p, T, branch=None):
        p = np.asarray(p, dtype=float)
        y1 = lp.y / (T * lp.alpha_S)
        y2, z2 = lp.y - y1, lp.z
        self.sell_pi = _ratio(y1, p)
        self.buy4 = _ratio(z2, p)
        self.sell4 = 1 - _ratio(y2, p)
        self.branch = branch

    def reset(self, N, rng):
        if self.branch is None:
            self.is_pi = rng.random(N) < 0.5
        else:
            self.is_pi = np.full(N, self.branch == 'pi')
        drop = rng.random(N) < 0.5  # Algoritmo 4, línea 2
        return np.where(self.is_pi | ~drop, 1, 0)

    def act(self, t, k, b, s, inv, rng):
        rho = rng.random(len(k))
        a = np.full(len(k), SKIP)
        pi, alg4 = self.is_pi, ~self.is_pi
        a[pi & (inv > 0) & (rho < self.sell_pi[k])] = SELL
        a[alg4 & (inv == 0) & (rho <= self.buy4[k]) & (self.buy4[k] > 0)] = BUY
        a[alg4 & (inv == 1) & (rho >= self.sell4[k]) & (self.sell4[k] < 1)] = SELL
        return a


def _ocrs_tables(qb, qs, pf, segments, m):
    """Probabilidades del Algoritmo 5 (OCRS) para cada segmento [l, r] de una solución con
    m-trades: se usa x/m, que es 1-trade. Devuelve tablas (T, K) para las requests 1..T."""
    T1, K = pf.shape
    prob_sell, prob_buy = np.zeros((T1, K)), np.zeros((T1, K))
    seg_start, in_seg = np.zeros(T1, dtype=bool), np.zeros(T1, dtype=bool)
    for l, r in segments:
        mm = max(1.0, qb[l:r + 1].sum()) if m == 'adaptive' else m
        for q, prob in ((qs, prob_sell), (qb, prob_buy)):
            q_seg = q[l:r + 1] / mm
            before = np.r_[0.0, np.cumsum(q_seg.sum(axis=1))[:-1]]  # sum_{j=l}^{i-1} q_j
            prob[l:r + 1] = _ratio(q_seg, pf[l:r + 1]) * (0.5 / (1 - 0.5 * before))[:, None]
        seg_start[l] = True
        in_seg[l:r + 1] = True
    return (np.clip(prob_sell[1:], 0, 1), np.clip(prob_buy[1:], 0, 1), seg_start[1:], in_seg[1:])


class Theorem12:
    """Teorema 1.2 (requests no idénticas, B = B0 = 1, ratio 1/22), a partir de la
    descomposición del Lema 2.5:
      rama 1 (prob. 1/11): x~(1) con una prophet inequality de un ítem (vender la copia inicial
                           con la OCRS de 1/2);
      ramas 2 y 3 (prob. 5/11 c/u): Corolario 2.4 sobre x~(2) o x~(3): conservar la copia con
                           prob. 1/2 y correr el Algoritmo 5 en cada segmento.
    m: 2.5 como en el paper, o 'adaptive' para usar m = max(1, masa del segmento)."""

    def __init__(self, dec, pf, m=2.5, probs=(1 / 11, 5 / 11, 5 / 11)):
        qs1 = dec.parts[0][1]
        before = np.r_[0.0, np.cumsum(qs1.sum(axis=1))[:-1]]
        self.prob_pi = np.clip(_ratio(qs1, pf) * (0.5 / (1 - 0.5 * before))[:, None], 0, 1)[1:]
        self.tables = [None] + [_ocrs_tables(*dec.parts[c], pf, dec.segments[c], m) for c in (1, 2)]
        self.probs = np.asarray(probs, dtype=float)

    def reset(self, N, rng):
        self.branch = rng.choice(3, size=N, p=self.probs / self.probs.sum())
        self.selling = np.zeros(N, dtype=bool)  # modo del segmento actual: vender o comprar
        self.done = np.zeros(N, dtype=bool)     # ya hizo su (única) transacción en el segmento
        keep = rng.random(N) < 0.5
        return np.where((self.branch == 0) | keep, 1, 0)

    def act(self, t, k, b, s, inv, rng):
        rho = rng.random(len(k))
        a = np.full(len(k), SKIP)
        a[(self.branch == 0) & (inv > 0) & (rho < self.prob_pi[t, k])] = SELL
        for c in (1, 2):
            prob_sell, prob_buy, seg_start, in_seg = self.tables[c]
            in_branch = self.branch == c
            if seg_start[t]:
                self.selling[in_branch] = inv[in_branch] > 0
                self.done[in_branch] = False
            if in_seg[t]:
                active = in_branch & ~self.done
                sell = active & self.selling & (rho < prob_sell[t, k])
                buy = active & ~self.selling & (rho < prob_buy[t, k])
                a[sell], a[buy] = SELL, BUY
                self.done |= sell | buy
        return a
