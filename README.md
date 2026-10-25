# Asymmetric Trading Prophets: simulaciones

Implementación computacional de Aggarwal, Gupta, Wang y Zhao (2026), *Asymmetric Trading Prophets* (`paper.pdf`).

## Cómo correr

```bash
python3 test_paper.py                  # chequea los lemas del paper (~1 s)
python3 experiments.py all --quick     # todos los experimentos en versión chica
python3 experiments.py all             # versión completa -> results/*.csv, results/*.png
python3 experiments.py large_asym      # uno en particular
```

Requiere `numpy`, `scipy`, `pandas`, `matplotlib` y `gurobipy`.

## Archivos

| Archivo | Contenido |
|---|---|
| `instances.py` | Distribuciones i.i.d., simétricas, no i.i.d. y Bernoulli; muestreo en lote; instancias de cota inferior (Sección 4) |
| `offline.py` | Profeta offline: DP exacta sobre (t, inventario), versión con política y versión vectorizada |
| `lp.py` | `solve_lp_iid` (LP IID, Sección 3.1) y `solve_lp_relax` (LP Relax, Sección 2 + Sección C) |
| `decomposition.py` | Uncrossing (Alg. 1), descomposición inicial (Alg. 2), separación x̃⁽¹⁾ + x̃⁽²⁾ + x̃⁽³⁾ (Lema 2.5) |
| `algorithms.py` | Políticas online: `Theorem12`, `Theorem13`, `Algorithm3`, `OnlineOptimal` |
| `simulate.py` | Simulador vectorizado, Monte Carlo con números aleatorios comunes, tabla resumen |
| `experiments.py` | Un experimento por resultado del paper |
| `plot.py` | Gráfico de precios, acciones e inventario de una trayectoria |
| `test_paper.py` | Tests de cada lema |

## Paper → código

| Resultado | Código | Experimento |
|---|---|---|
| Thm 1.1 (B₀ = 0 imposible) | `lb_instance_thm11` + `OnlineOptimal` | `lower_bounds` |
| Thm 1.2 (no i.i.d., 1/22) | `solve_lp_relax` → `decompose` → `Theorem12` | `noniid` |
| Thm 1.3 (i.i.d., B = 1, 1/4) | `solve_lp_iid` → `Theorem13` (Lema 3.4 + Alg. 4) | `unit_iid` |
| Thm 1.4(a) (asimétrico, 1 − O(log B₀/√B₀)) | `Algorithm3(cap=B0)` | `large_asym` |
| Thm 1.4(b) (1 − Ω(1/√B₀)) | `lb_instance_thm14b` + `OnlineOptimal` | `lower_bounds` |
| Thm 1.5(a) (simétrico, 1 − O(log B/√B)) | `Algorithm3(cap=B, symmetric=True)` | `large_sym` |
| Thm 1.5(b) (1 − Ω(1/√B)) | `lb_instance_thm15b` + `OnlineOptimal` | `lower_bounds` |

## Decisiones de implementación

- **Uncrossing (Alg. 1).** Su salida no tiene pares anidados. Eso equivale al acoplamiento FIFO ("first buy first sell") de las masas q^b y q^s en el tiempo, que es único, así que se construye directo con una cola. `test_uncross_equals_algorithm1` lo compara con el Algoritmo 1 literal.
- **Restricción (2) de LP Relax.** Se escribe como H_t = H_{t−1} + q^b_{t−1} − q^s_{t−1} ≤ 1, que es equivalente y evita O(T³) coeficientes.
- **Tipos dentro de un trade (i, j).** Después del uncrossing (Sección C), se reparten proporcionalmente a q^s_{j,k}. El paper sólo necesita que se preserven q^b y q^s.
- **"Botar la copia inicial"** (Alg. 4 y Cor. 2.4). El modelo no tiene esa acción, así que se implementa como descarte gratuito en t = 0 (`reset` devuelve cuántas copias se conservan).
- **Algoritmo 3.** La constante de τ = T − ⌈c log B √(T/α_S)⌉ es un parámetro (el paper usa c = 13). Con c = 13, τ = 0 en todos los tamaños razonables y el algoritmo sólo vende. También se aplica B ← min(B, 2Γ), como en la demostración del Teorema 3.2.
- **Agregados que no están en el paper:**
  - `OnlineOptimal`: la política online óptima exacta por DP, como referencia.
  - `Theorem12(m='adaptive')`: escala cada segmento por su propia masa en vez de 2.5.
  - `lb_instance_thm14b`: instancia concreta para el Thm 1.4(b), donde el paper sólo cita [HKS07]. Está armada como la del Thm 1.5(b).
