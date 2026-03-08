import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import random
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


def estimate_all_coef(serie, points):
    coef = []
    start = 0
    pts = points.copy()
    pts.append(len(serie))

    for end in pts:
        x = list(range(start, end))
        y = serie[start:end]
        coef.append(estimate_segment_coef(x, y))
        start = end

    return coef


def estimate_all_points(coef, points, n):
    estimated = []
    start = 0
    pts = points.copy()
    pts.append(n)

    for idx, end in enumerate(pts):
        m, b = coef[idx]
        for i in range(start, end):
            estimated.append(m * i + b)
        start = end

    return estimated


# ============================================================
# 3. GENERACIÓN DE PUNTOS ALEATORIOS
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
# 4. CÁLCULO DEL RMSE
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
# 5. RANDOM SEARCH
# ============================================================

def randomSearch(serie, k, n_iter, prev_best=None):

    if prev_best is None:
        best = {'rmse': float('inf'), 'points': None}
    else:
        best = prev_best

    startTime = time.time()
    rmses = []

    for _ in range(n_iter):
        points = randomPoints(len(serie), k)
        rmse = mean_rmse(serie, points)

        if rmse < best['rmse']:
            best['rmse'] = rmse
            best['points'] = points
            rmses.append(best.copy())

    endTime = time.time()
    i_time = endTime - startTime

    return best, i_time, rmses


# ============================================================
# 6. EJECUCIÓN GLOBAL DEL EXPERIMENTO
# ============================================================

def ejecutar_RS(series, k_values, start=10, end=200, step=10, repeticiones=20):
    resultados = []
    print("\n === Ejecutando experimento Random Search === ")

    for rep in range(repeticiones):
        print(f"Repetición {rep+1}/{repeticiones} completada")
        rep_result = []

        for idx, serie in enumerate(series):
            k = k_values[idx]
            best = None
            serie_result = []

            for n in range(start, end, step):
                best, tiempo, historial = randomSearch(serie, k, step, best)
                serie_result.append({
                    "iter": n,
                    "rmse": best["rmse"],
                    "points": best["points"],
                    "tiempo": tiempo
                })

            rep_result.append(serie_result)

        resultados.append(rep_result)

    print(" === Experimento finalizado === \n")
    return resultados


# ============================================================
# 7. MÉTRICAS
# ============================================================

def exactitud_RS(resultados):
    medias = []
    for serie in range(len(resultados[0])):
        rmse_final = [rep[serie][-1]["rmse"] for rep in resultados]
        medias.append(np.mean(rmse_final))
    return medias


def variabilidad_RS(resultados):
    desv = []
    for serie in range(len(resultados[0])):
        rmse_final = [rep[serie][-1]["rmse"] for rep in resultados]
        desv.append(np.std(rmse_final))
    return desv


def tiempo_RS(resultados):
    tiempos = []
    for serie in range(len(resultados[0])):
        tiempo_total = [sum([it["tiempo"] for it in rep[serie]]) for rep in resultados]
        tiempos.append(np.mean(tiempo_total))
    return tiempos


# ============================================================
# 8. GRÁFICAS
# ============================================================

def plot_RS(series, resultados):
    print(">>> Generando gráficas de Búsqueda Aleatoria ... ")
    num_series = len(series)

    for idx in range(num_series):
        rmse_iters = np.array([
            [it["rmse"] for it in rep[idx]]
            for rep in resultados
        ])

        mean_rmse = rmse_iters.mean(axis=0)
        std_rmse = rmse_iters.std(axis=0)
        iters = [it["iter"] for it in resultados[0][idx]]

        plt.figure(figsize=(12, 5))
        plt.fill_between(iters, mean_rmse - std_rmse, mean_rmse + std_rmse,
                         color="#4A90E2", alpha=0.20, label="+1 desviación")

        plt.plot(iters, mean_rmse, color="#1F4E79", linewidth=2.2,
                 marker="o", markersize=5, label="Media RMSE")

        plt.scatter(iters[0], mean_rmse[0], color="green", s=80, label="Inicio")
        plt.scatter(iters[-1], mean_rmse[-1], color="red", s=80, label="Fin")

        plt.title(f"Evolución del RMSE - Random Search (TS{idx+1})", fontsize=14)
        plt.xlabel("Iteraciones")
        plt.ylabel("RMSE")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"TS{idx+1}_mean.png", dpi=150)

        plt.show()

    print(">>> Gráficas generadas correctamente.")



def plot_final_solutions(series, resultados, k_values):
    for idx, serie in enumerate(series):
        best_points = resultados[-1][idx][-1]["points"]

        coef = estimate_all_coef(serie, best_points.copy())
        estimada = estimate_all_points(coef, best_points.copy(), len(serie))

        plt.figure(figsize=(12, 5))
        plt.plot(serie, label="Serie real", color="blue")
        plt.plot(estimada, label="Serie estimada", color="red")

        for p in best_points:
            plt.axvline(x=p, color="black", linestyle="--", alpha=0.7)

        plt.title(f"Solución final RS - TS{idx+1} (k={k_values[idx]})")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"RS_TS{idx+1}_solucion_final.png", dpi=150)

        plt.show()

