import numpy as np
from instances import generate_basic_instance

# Acciones codificadas como el cambio de inventario que producen
SKIP, BUY, SELL = 0, 1, -1


def offline_opt(b, s, B, B_0):
    T = len(b)
    V = np.full((T + 1, B + 1), -np.inf)
    V[T, :] = 0.0 # Al final no vale nada el activo

    policy = np.zeros((T, B + 1), dtype=int)  # policy[t, x] = acción óptima en (t, x)

    for t in range(T - 1, -1, -1):
        for x in range(B + 1):
            best, act = V[t + 1, x], SKIP                    # skip
            if x < B and V[t + 1, x + 1] - b[t] > best:      # buy
                best, act = V[t + 1, x + 1] - b[t], BUY
            if x > 0 and V[t + 1, x - 1] + s[t] > best:      # sell
                best, act = V[t + 1, x - 1] + s[t], SELL
            V[t, x] = best
            policy[t, x] = act
    return V[0, B_0], V, policy


def recover_path(policy, B_0):
    """Recorre la política hacia adelante desde x = B_0 y devuelve
    las acciones tomadas y el inventario (inventory[t] = inventario tras t requests)."""
    T = policy.shape[0]
    actions = np.zeros(T, dtype=int)
    inventory = np.zeros(T + 1, dtype=int)
    inventory[0] = x = B_0
    for t in range(T):
        actions[t] = policy[t, x]
        x += actions[t]
        inventory[t + 1] = x
    return actions, inventory


if __name__ == "__main__":
    instance = generate_basic_instance(T=10)
    b = [b for b, s in instance]
    s = [s for b, s in instance]
    B = 1
    B_0 = 1

    optimal_value, V, policy = offline_opt(b, s, B, B_0)
    actions, inventory = recover_path(policy, B_0)
    names = {SKIP: 'skip', BUY: 'buy', SELL: 'sell'}
    print("Optimal value:", optimal_value)
    print("Value function V:\n", V)
    for t, a in enumerate(actions):
        print(f"Time: {t + 1}, Inventory before: {inventory[t]}, Action: {names[a]}")
