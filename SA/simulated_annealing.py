import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
import math
import time


# ============================================================
# 1. CARGA DE SERIES
# ============================================================


def read_serie(path):
    with open(path, "r") as f:
        contenido = f.read().replace("[", "").replace("]", "").split()
    return [float(x) for x in contenido]


# ============================================================
# 2. REGRESIÓN LINEAL POR SEGMENTOS
# ============================================================


def estimate_segment_coef(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    if len(x) < 2:
        return (0.0, 0.0)
    model = LinearRegression().fit(x, y)
    return (model.coef_[0], model.intercept_)

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
# 3. PUNTOS ALEATORIOS
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
# 4. RMSE
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
# 5. GENERACIÓN DE VECINOS
# ============================================================


def random_neighbour(points, n):
    vecinos = []

    for i, point in enumerate(points):
        # mover + 1% del tamaño de la serie
        v1 = points.copy()
        v1[i] = point + int(n / 50)
        vecinos.append(v1)

        # mover - 1% del tamaño de la serie
        v2 = points.copy()
        v2[i] = point - int(n / 50)
        vecinos.append(v2)

    # Filtrar vecinos inválidos
    validos = []
    for v in vecinos:
        if v[0] <= 0:
            continue
        if v[-1] >= n:
            continue
        if sorted(v) != v:
            continue
        if len(set(v)) != len(v):
            continue
        validos.append(v)

    return random.choice(validos)


# ============================================================
# 6. SIMULATED ANNEALING
# ============================================================


def simulated_annealing(T0, funcion_enfriamiento, p, L, Tf, serie, k, prev_best=None):
    startTime = time.time()

    T = T0
    sol = randomPoints(len(serie), k)
    rmse_sol = mean_rmse(serie, sol)
    best = sol.copy()
    rmse_best = rmse_sol
    iter = 1

    while T >= Tf:
        for _ in range(L):
            vecino = random_neighbour(sol, len(serie))
            rmse_vecino = mean_rmse(serie, vecino)

            delta = rmse_vecino - rmse_sol

            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / max(T, 1e-8)):
                sol = vecino
                rmse_sol = rmse_vecino

                if rmse_sol < rmse_best:
                    best = sol.copy()
                    rmse_best = rmse_sol

        T = funcion_enfriamiento(T, T0, iter, p)
        iter += 1

    if  prev_best!=None and rmse_best > prev_best["rmse"]:
        best=prev_best["points"]
        rmse_best=prev_best["rmse"]

    endTime = time.time()
    return {"rmse": rmse_best, "points": best}, endTime - startTime


# ============================================================
# 7. FUNCIONES DE ENFRIAMIENTO
# ============================================================

enfriamiento_lineal = lambda T, T0, i, B: T0 - (i * B)
enfriamiento_geometrico = lambda T, T0, i, param: T * param
enfriamiento_logaritmico = lambda T, T0, i, B: T0 / (1 + B*math.log(i))


# ============================================================
# 8. EJECUCIÓN DEL EXPERIMENTO
# ============================================================


def ejecutar_experimento(
    series,
    k_values,
    nombre,
    funcion_enfriamiento,
    p=None,
    T0=100,
    Tf=0.1,
    start=10,
    end=100,
    increment=10,
):
    print(f"\n=== Ejecutando SA con enfriamiento {nombre} ===")

    mean_iters = []
    mean_bests = []
    mean_times = []

    curr_best=[]
    for i in range(20):
        series_iters = []
        series_bests = []
        series_times = []
        for TS in range(len(series)):
            times = []
            iters = []
            bests = []

            current_serie = series[TS]
            current_k = k_values[TS]
            
            if(i==0):
                best, time_exec = simulated_annealing(
                    T0=T0,
                    funcion_enfriamiento=funcion_enfriamiento,
                    L=50,
                    Tf=Tf,
                    serie=current_serie,
                    k=current_k,
                    p=p
                )
                curr_best.append(best)
            else:
                best, time_exec = simulated_annealing(
                    T0=T0,
                    funcion_enfriamiento=funcion_enfriamiento,
                    L=50,
                    Tf=Tf,
                    serie=current_serie,
                    k=current_k,
                    p=p,
                    prev_best=curr_best[TS]
                )
                curr_best[TS]=best

            series_iters.append(i)
            series_bests.append(best)
            series_times.append(time_exec)

        mean_iters.append(series_iters)
        mean_bests.append(series_bests)
        mean_times.append(series_times)

    print("=== Experimento finalizado ===")
    return mean_iters, mean_bests, mean_times


# ============================================================
# 9. GRÁFICAS (GUARDAR PNG)
# ============================================================


def plot_SA(mean_bests, mean_iters, nombre):
    mean_bests = np.array(mean_bests)
    mean_iters = np.array(mean_iters)

    for TS in range(len(mean_bests[0])):
        plt.figure(figsize=(10, 5))
        plt.title(f"SA - Evolución RMSE TS{TS + 1} ({nombre})")

        iters = mean_iters[0][TS]
        rmses = [rmse[TS]["rmse"] for rmse in mean_bests]
        plt.plot(rmses, label="RMSE", color="blue",marker="o")

        plt.xlabel("iters")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()

        plt.savefig(f"SA_TS{TS + 1}_evolucion_RMSE_{nombre}.png", dpi=150)
        plt.show()


def plot_final_SA(series, resultados, titulo):
    for TS in range(len(series)):
        plt.figure(figsize=(12, 5))
        plt.title(f"SA - Solución final {titulo} - TS{TS + 1}")

        plt.plot(series[TS], label="Serie real", color="blue")

        best_points = resultados[-1][TS][-1]["points"]

        for p in best_points:
            plt.axvline(x=p, linestyle="--", color="black")

        coef = estimate_all_coef(series[TS], best_points)
        estimada = estimate_all_points(coef, best_points, len(series[TS]))

        plt.plot(estimada, label="Serie estimada", color="red")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"SA_TS{TS + 1}_solucion_final_{titulo}.png", dpi=150)
        plt.show()
