from collections import deque
from dataclasses import dataclass
import numpy as np
from lp import lp_value

TOL = 1e-9


def uncross(qb, qs):
    """Algoritmo 1 (uncrossing). Su salida no tiene pares i < i' < j' < j con x_ij, x_i'j' > 0,
    o sea, es el acoplamiento "first buy first sell" (FIFO) entre las masas de compra y de
    venta en el tiempo. Ese acoplamiento es único y sólo depende de q^b y q^s (que el
    uncrossing no cambia), así que se construye directamente con una cola.

    Devuelve X (T+1, T+1): X[i, j] = masa (sumada sobre tipos) comprada en i y vendida en j."""
    QB, QS = qb.sum(axis=1), qs.sum(axis=1)
    X = np.zeros((len(QB), len(QB)))
    queue = deque()  # compras aún no vendidas: [tiempo, masa]
    for t in range(len(QB)):
        need = QS[t]
        while need > TOL and queue:
            i, m = queue[0]
            used = min(need, m)
            X[i, t] += used
            need -= used
            if m - used > TOL:
                queue[0][1] = m - used
            else:
                queue.popleft()
        if need > 1e-6:
            raise ValueError(f'q^b, q^s no factibles: faltan {need:.2e} de compras antes de t={t}')
        if QB[t] > TOL:
            queue.append([t, QB[t]])
    return X


def initial_decomposition(QB, QS):
    """Algoritmo 2: parte [1, T] en intervalos [l, r], cerrando cada uno en el primer r con
    sum_{l..r} q^b >= 1 y sum_{l..r} q^s >= 1 (el último termina en T)."""
    T = len(QB) - 1
    intervals, l = [], 1
    while l <= T:
        ok = np.nonzero((np.cumsum(QB[l:]) >= 1 - TOL) & (np.cumsum(QS[l:]) >= 1 - TOL))[0]
        r = l + ok[0] if len(ok) else T
        intervals.append((l, r))
        l = r + 1
    return intervals


@dataclass
class Decomposition:
    X: np.ndarray       # acoplamiento sin cruces (salida de uncross)
    intervals: list     # salida del Algoritmo 2
    parts: list         # [(qb, qs)] de x~(1), x~(2), x~(3)
    segments: list      # [None, segmentos de x~(2), segmentos de x~(3)]
    values: list        # LP(x~(1)), LP(x~(2)), LP(x~(3))


def decompose(lp):
    """Lema 2.5: x* -> x~ (uncrossing) -> x~(1) + x~(2) + x~(3).

    x~(1): trades que compran en la request 0 (la copia inicial).
    x~(2): trades que compran en un intervalo impar del Algoritmo 2; x~(3): en uno par.
    Por el Lema 2.7.(ii) cada trade de x~(2) vive en [l_d, r_{d+1}] con d impar: esos son
    sus segmentos (disjuntos) para el Corolario 2.4, y análogamente para x~(3)."""
    qb, qs = lp.qb, lp.qs
    X = uncross(qb, qs)
    QB, QS = qb.sum(axis=1), qs.sum(axis=1)
    intervals = initial_decomposition(QB, QS)

    interval_of = np.zeros(len(QB), dtype=int)  # d (1-indexado) de cada request; 0 = request 0
    for d, (l, r) in enumerate(intervals, 1):
        interval_of[l:r + 1] = d
    groups = [interval_of == 0,
              interval_of % 2 == 1,
              (interval_of > 0) & (interval_of % 2 == 0)]

    # Dentro de cada (i, j) los tipos se reparten proporcionalmente a q^s_{j,k} / q^s_j
    w_sell = np.divide(qs, QS[:, None], out=np.zeros_like(qs), where=QS[:, None] > TOL)
    parts = [(qb * g[:, None], X[g].sum(axis=0)[:, None] * w_sell) for g in groups]

    D = len(intervals)
    def union(d):  # [l_d, r_{d+1}] (o [l_D, r_D] si d es el último), d 0-indexado
        return (intervals[d][0], intervals[min(d + 1, D - 1)][1])
    segments = [None, [union(d) for d in range(0, D, 2)], [union(d) for d in range(1, D, 2)]]

    values = [lp_value(qb_c, qs_c, lp.bf, lp.sf) for qb_c, qs_c in parts]
    return Decomposition(X=X, intervals=intervals, parts=parts, segments=segments, values=values)


def check_lemma_2_7(dec):
    """Devuelve (máx. masa de compra/venta por intervalo, máx. # de intervalos que cruza un
    trade con i != 0). El Lema 2.7 garantiza <= 2.5 y <= 1."""
    QB, QS = dec.X.sum(axis=1), dec.X.sum(axis=0)
    max_mass = max(max(QB[l:r + 1].sum(), QS[l:r + 1].sum()) for l, r in dec.intervals)
    interval_of = np.zeros(len(QB), dtype=int)
    for d, (l, r) in enumerate(dec.intervals, 1):
        interval_of[l:r + 1] = d
    I, J = np.nonzero(dec.X[1:] > TOL)
    max_cross = int((interval_of[J] - interval_of[I + 1]).max()) if len(I) else 0
    return max_mass, max_cross
