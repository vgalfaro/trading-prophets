import numpy as np

def generate_basic_instance(T: int):

    # 1. Generar el precio base (ej. el Bid) usando Lognormal
    b_t = np.random.lognormal(mean=4.0, sigma=0.3, size=T)

    # 2. Generar un 'spread' (diferencia) siempre positivo usando una distribución Exponencial
    # La distribución exponencial es excelente para modelar tiempos de espera o márgenes pequeños >= 0
    spread_t = np.random.exponential(scale=1.5, size=T)

    # 3. El precio de venta (Ask/Sell) es el precio base más el spread
    s_t = b_t + spread_t

    tuplas_bid_ask = [(round(b, 2), round(s, 2)) for b, s in zip(b_t, s_t)]

    # print("\nPrecios con spread asegurado (b_t < s_t):")
    # print(tuplas_bid_ask[:5])

    return tuplas_bid_ask

if __name__ == "__main__":
    generate_basic_instance(T=10)