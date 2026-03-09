import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
import time


# ============================================================
# 1. REGRESIÓN LINEAL POR SEGMENTOS
# ============================================================


def read_serie(path):
    with open(path, "r") as f:
        contenido = f.read().replace("[", "").replace("]", "").split()
    return [float(x) for x in contenido]



def estimate_coef(x, y):
    x = np.array(x)
    y = np.array(y)
    n = np.size(x)

    if n == 0:
        return (0.0, 0.0)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    if abs(SS_xx) < 1e-8:
        b_0 = 0.0
        b_1 = m_y
    else:
        b_0 = SS_xy / SS_xx
        b_1 = m_y - b_0 * m_x

    return (float(b_0), float(b_1))


def estimate_all_coef(temp, points):
    coef = []
    start = 0

    pts = points.copy()
    pts.append(len(temp))

    for point in pts:
        b_0, b_1 = estimate_coef(range(start, point), temp[start:point])
        coef.append((b_0, b_1))
        start = point

    return coef


def estimate_point(coef, i):
    return i * coef[0] + coef[1]

def estimate_all_points(coef, points, temp_size):
    estimated = []
    start = 0

    pts = points.copy()
    pts.append(temp_size)

    for pos, point in enumerate(pts):
        for i in range(start, point):
            estimated.append(estimate_point(coef[pos], i))
        start = point

    return estimated


# ============================================================
# 2. PUNTOS ALEATORIOS
# ============================================================


def randomPoints(longitud_serie, n_cortes):
    b = []
    while n_cortes > 0:
        r = random.randint(1, longitud_serie - 1)
        if r not in b:
            b.append(r)
            n_cortes -= 1
    b.sort()
    return b


# ============================================================
# 3. RMSE
# ============================================================


def RMSE(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    return np.sqrt(mse)


def mean_rmse(serie, points):
    rmse_acc = 0.0
    start = 0
    coef = estimate_all_coef(serie, points.copy())
    estimated = estimate_all_points(coef, points.copy(), len(serie))
    pts = points.copy()
    pts.append(len(serie))

    for point in pts:
        rmse_acc += RMSE(serie[start:point], estimated[start:point])
        start = point

    return rmse_acc / len(pts)


# ============================================================
# 4. GENERACIÓN DE VECINOS
# ============================================================


def search_neighbour(serie, n):
    """
    Genera todos los vecinos válidos moviendo un punto ±1.
    Filtra duplicados, desordenados y puntos fuera de rango.
    """
    vecinos = []

    for i, point in enumerate(serie):
        # Mover + 1% del tamaño de la serie
        v1 = serie.copy()
        v1[i] = point + int(n / 50)
        vecinos.append(v1)

        # Mover - 1% del tamaño de la serie
        v2 = serie.copy()
        v2[i] = point - int(n / 50)
        vecinos.append(v2)

    validos = []

    for vect in vecinos:
        if vect[0] <= 0:
            continue
        if vect[-1] >= n:
            continue
        if len(set(vect)) != len(vect):
            continue
        if vect != sorted(vect):
            continue

        validos.append(vect)

    return validos


# ============================================================
# 5. HILL CLIMBING
# ============================================================


def hc(serie, k):
    startTime = time.time()

    solucion = randomPoints(len(serie), k)
    rmse = mean_rmse(serie, solucion)
    mejor_vecino = []
    evolution=[]
    while mejor_vecino is not None:
        vecinos = search_neighbour(solucion, len(serie))

        mejor_vecino = None 
        mejor_rmse = rmse

        for v in vecinos:
            current = mean_rmse(serie, v)
            if current < mejor_rmse:
                mejor_rmse = current
                mejor_vecino = v

        if mejor_vecino is not None:
            solucion = mejor_vecino
            rmse = mejor_rmse

        evolution.append({"rmse":rmse,"points":solucion})


    endTime = time.time()
    return {"rmse": rmse, "points": solucion}, endTime - startTime, evolution


# ============================================================
# 6. EJECUCIÓN DEL EXPERIMENTO
# ============================================================


def ejecutar_HC(series, k_values, repeticiones=20):
    resultados = []

    print("\n=== Ejecutando experimento Hill Climbing ===")

    for idx, serie in enumerate(series):
        k = k_values[idx]
        print(f"\nProcesando TS{idx + 1} (k={k})...")

        rep_result = []
        for rep in range(repeticiones):
            sol, tiempo, _= hc(serie, k)

            rep_result.append(
                {"rmse": sol["rmse"], "points": sol["points"], "tiempo": tiempo}
            )
            print(f"Repetición {rep + 1}/{repeticiones} completada")

        resultados.append(rep_result)

    print("\n=== Experimento finalizado ===")
    return resultados


# ============================================================
# 7. GRÁFICAS
# ============================================================


def plot_HC_RMSE(series, resultados):
    for idx in range(len(series)):
        rmse_vals = [rep["rmse"] for rep in resultados[idx]]
        ejecuciones = list(range(1, len(rmse_vals) + 1))

        mean_rmse_val = np.mean(rmse_vals)
        std_rmse_val = np.std(rmse_vals)

        plt.figure(figsize=(10, 5))
        plt.title(f"Evolución del RMSE - Hill Climbing (TS{idx + 1})")

        plt.plot(
            ejecuciones, rmse_vals, marker="o", color="blue", label="RMSE por ejecución"
        )
        plt.hlines(
            mean_rmse_val,
            1,
            len(rmse_vals),
            color="green",
            linewidth=2,
            label="Media RMSE",
        )

        plt.fill_between(
            ejecuciones,
            mean_rmse_val - std_rmse_val,
            mean_rmse_val + std_rmse_val,
            color="green",
            alpha=0.2,
            label="±1 desviación",
        )

        plt.xlabel("Ejecución")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"TS{idx + 1}_mean.png", dpi=150)
        plt.show()


def plot_final_HC(series, resultados, k_values):
    for idx, serie in enumerate(series):
        best = min(resultados[idx], key=lambda x: x["rmse"])
        points = best["points"]

        coef = estimate_all_coef(serie, points)
        estimada = estimate_all_points(coef, points, len(serie))

        plt.figure(figsize=(12, 5))
        plt.plot(serie, label="Serie real", color="blue")
        plt.plot(estimada, label="Serie estimada", color="red")

        for p in points:
            plt.axvline(x=p, linestyle="--", color="black")

        plt.title(f"Solución final HC - TS{idx + 1} (k={k_values[idx]})")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"TS{idx + 1}_sol.png", dpi=150)
        plt.show()
