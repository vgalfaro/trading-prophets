import numpy as np
import matplotlib.pyplot as plt
from instances import generate_basic_instance
from offline import offline_opt, recover_path, BUY, SELL


def plot_instance(b, s, actions, inventory, B, title=None):
    b, s = np.asarray(b, dtype=float), np.asarray(s, dtype=float)
    T = len(b)
    t = np.arange(1, T + 1)
    buy, sell = actions == BUY, actions == SELL

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6),
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Precios: b_t y s_t (los inf, i.e. "no se puede comprar", no se dibujan)
    ax1.plot(t, b, "-o", ms=3, lw=1, color="tab:blue", label="$b_t$ (costo de comprar)")
    ax1.plot(t, s, "-o", ms=3, lw=1, color="tab:orange", label="$s_t$ (ingreso por vender)")

    # Acciones del profeta: compra sobre b_t, venta sobre s_t
    ax1.scatter(t[buy], b[buy], marker="^", s=130, color="green", zorder=3, label="compra")
    ax1.scatter(t[sell], s[sell], marker="v", s=130, color="red", zorder=3, label="venta")
    ax1.set_ylabel("precio")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)
    if title:
        ax1.set_title(title)

    # Inventario: inventory[k] es el inventario después de la request k (k=0: inicial)
    ax2.step(np.arange(T + 1), inventory, where="post", color="black")
    ax2.set_ylim(-0.3, B + 0.3)
    ax2.set_ylabel("inventario")
    ax2.set_xlabel("t")
    ax2.grid(alpha=0.3)
    if B <= 10:
        ax2.set_yticks(range(B + 1))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    T, B, B_0 = 30, 1, 1

    instance = generate_basic_instance(T=T)
    b = np.array([b for b, s in instance])
    s = np.array([s for b, s in instance])

    opt, V, policy = offline_opt(b, s, B, B_0)
    actions, inventory = recover_path(policy, B_0)

    # Chequeo: la ganancia de la trayectoria recuperada debe coincidir con V[0, B_0]
    profit = s[actions == SELL].sum() - b[actions == BUY].sum()
    assert np.isclose(profit, opt), (profit, opt)

    fig = plot_instance(b, s, actions, inventory, B,
                        title=f"Profeta offline  (T={T}, B={B}, B0={B_0}, OPT={opt:.2f})")
    fig.savefig("offline_path.png", dpi=150)
    plt.show()
