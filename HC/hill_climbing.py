import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# 5. GENERACIÓN DE VECINOS
# ============================================================

def search_neighbour(serie, n):
    v = []

    for i, point in enumerate(serie):
        serie_aux = serie.copy()
        serie_aux[i] = point+1
        v.append(serie_aux)

        serie_aux = serie.copy()
        serie_aux[i] = point-1
        v.append(serie_aux)

    p = []

    for vect in v:
        if vect[0] == 0:
            continue
        if vect[len(vect)-1]==n:
            continue
        flag = False
        for i in range(len(vect)-1):
            if vect[i] == vect[i+1]:
                flag = True
                break
        if flag:
            continue
        
        p.append(vect)

    return p

# ============================================================
# 6. ALGORITMO HILL CLIMBING
# ============================================================

def hc(serie, k):
    startTime = time.time()
    solucion = randomPoints(len(serie),k)
    rmse = mean_rmse(serie, solucion)
    flag = True

    while(flag):
        flag = False
        v = search_neighbour(solucion, len(serie))

        for vect in v:
            current = mean_rmse(serie, vect)

            if current < rmse:
                solucion = vect
                rmse = current
                flag = True
                break

    endTime = time.time()
    t = endTime - startTime

    return {'rmse': rmse, 'points': solucion}, t


# ============================================================
# 7. EJECUCIÓN DEL EXPERIMENTO
# ============================================================

def ejecutar_HC(series, k_values, repeticiones=20):
    resultados = []

    print("\n=== Ejecutando experimento Hill Climbing ===")

    for idx, serie in enumerate(series):
        k = k_values[idx]
        print(f"\nProcesando TS{idx+1} (k={k}) ...")

        rep_result = []

        for rep in range(repeticiones):
            sol, tiempo = hc(serie, k)
            rep_result.append({
                "rmse": sol["rmse"],
                "points": sol["points"],
                "tiempo": tiempo
            })
            print(f"Repetición {rep+1}/{repeticiones} completada")

        resultados.append(rep_result)

    print("\n=== Experimento finalizado ===")
    return resultados


# ============================================================
# 8. GRÁFICAS (GUARDADAS EN PNG)
# ============================================================

def plot_HC_RMSE(series, resultados):
    for idx in range(len(series)):
        rmse_vals = [rep["rmse"] for rep in resultados[idx]]
        ejecuciones = list(range(1, len(rmse_vals) + 1))

        mean_rmse_val = np.mean(rmse_vals)
        std_rmse_val = np.std(rmse_vals)

        plt.figure(figsize=(10, 5))
        plt.title(f"Evolución del RMSE - Hill Climbing (TS{idx+1})")

        plt.plot(ejecuciones, rmse_vals, marker="o", color="blue", label="RMSE por ejecución")
        plt.hlines(mean_rmse_val, 1, len(rmse_vals), color="green", linewidth=2, label="Media RMSE")

        plt.fill_between(
            ejecuciones,
            mean_rmse_val - std_rmse_val,
            mean_rmse_val + std_rmse_val,
            color="green",
            alpha=0.2,
            label="+1 desviación"
        )

        plt.xlabel("Ejecución")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()

        plt.savefig(f"HC_TS{idx+1}_evolucion_RMSE.png", dpi=150)

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

        plt.title(f"Solución final HC - TS{idx+1} (k={k_values[idx]})")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"HC_TS{idx+1}_solucion_final.png", dpi=150)

        plt.show()

