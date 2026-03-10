from enum import Flag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
import math
import time
from tabulate import tabulate


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
    neighbour = []
    flag = True

    while flag:
        neighbour = points.copy()
        neighbour[random.randint(0, len(neighbour) - 1)] += random.choice(
            (-1, 1)
        ) * int(n / 50)

        if neighbour[0] <= 0:
            continue
        if neighbour[-1] >= n:
            continue
        if sorted(neighbour) != neighbour:
            continue
        if len(set(neighbour)) != len(neighbour):
            continue
        flag = False

    return neighbour


# ============================================================
# 6. SIMULATED ANNEALING
# ============================================================


def simulated_annealing(T0, funcion_enfriamiento, p, L, Tf, serie, k):
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

    endTime = time.time()
    return {"rmse": rmse_best, "points": best}, endTime - startTime


# ============================================================
# 7. FUNCIONES DE ENFRIAMIENTO
# ============================================================

enfriamiento_lineal = lambda T, T0, i, p: T0 - (i * p)
enfriamiento_exponencial = lambda T, T0, i, p: T0 * p**i
enfriamiento_logaritmico = lambda T, T0, i, p: T0 / (1 + math.log(i))


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
):
    print(f"\n=== Ejecutando SA con enfriamiento {nombre} ===")

    mean_bests = []
    mean_times = []

    for i in range(20):
        series_bests = []
        series_times = []
        for TS in range(len(series)):

            current_serie = series[TS]
            current_k = k_values[TS]


            best, time_exec = simulated_annealing(
                T0=T0,
                funcion_enfriamiento=funcion_enfriamiento,
                L=50,
                Tf=Tf,
                serie=current_serie,
                k=current_k,
                p=p)

            series_bests.append(best)
            series_times.append(time_exec)
        print(f"Repetición {i + 1}/{20} completada")

        mean_bests.append(series_bests)
        mean_times.append(series_times)

    print("=== Experimento finalizado ===")
    return mean_bests, mean_times


def ejecutar_experimento_p(
    series,
    k_values,
    nombre,
    funcion_enfriamiento,
    pi=1,
    pf=200,
    step=20,
    T0=100,
    Tf=0.1,
    L=20,
):
    print(f"\n=== Ejecutando SA con enfriamiento {nombre} ===")

    mean_bests = []
    mean_times = []
    p=[]
    for i in range(20):
        series_bests = []
        series_times = []
        for TS in range(len(series)):

            current_serie = series[TS]
            current_k = k_values[TS]
            
            bests=[]
            times=[]
            for p_val in range(pi,pf,step):
                best, time_exec = simulated_annealing(
                    T0=T0,
                    funcion_enfriamiento=funcion_enfriamiento,
                    L=L,
                    Tf=Tf,
                    serie=current_serie,
                    k=current_k,
                    p=p_val)
                bests.append(best)
                times.append(time_exec)
                if TS==0 and i==0:
                    p.append(p_val)

            series_bests.append(bests)
            series_times.append(times)
        print(f"Repetición {i + 1}/{20} completada")

        mean_bests.append(series_bests)
        mean_times.append(series_times)

    print("=== Experimento finalizado ===")
    return mean_bests, mean_times,p

def ejecutar_experimento_L(
    series,
    k_values,
    nombre,
    funcion_enfriamiento,
    Li=10,
    Lf=100,
    p=200,
    step=20,
    T0=100,
    Tf=0.1,
    L=20,
):
    print(f"\n=== Ejecutando SA con enfriamiento {nombre} ===")

    mean_bests = []
    mean_times = []
    L=[]
    for i in range(20):
        series_bests = []
        series_times = []
        for TS in range(len(series)):

            current_serie = series[TS]
            current_k = k_values[TS]
            
            bests=[]
            times=[]
            for L_val in range(Li,Lf,step):
                best, time_exec = simulated_annealing(
                    T0=T0,
                    funcion_enfriamiento=funcion_enfriamiento,
                    L=L_val,
                    Tf=Tf,
                    serie=current_serie,
                    k=current_k,
                    p=p)
                bests.append(best)
                times.append(time_exec)
                if TS==0 and i==0:
                    L.append(L_val)

            series_bests.append(bests)
            series_times.append(times)
        print(f"Repetición {i + 1}/{20} completada")

        mean_bests.append(series_bests)
        mean_times.append(series_times)

    print("=== Experimento finalizado ===")
    return mean_bests, mean_times,L


def SA_Lambda(series, k_values):
    bests_exp, times_exp = ejecutar_experimento(
    series, k_values, "exponencial", enfriamiento_exponencial, p=0.5, Tf=20
    )

    bests_log, times_log = ejecutar_experimento(
    series, k_values, "logarítmico", enfriamiento_logaritmico, Tf=20
    )


    bests_lin, times_lin = ejecutar_experimento(
    series, k_values, "lineal", enfriamiento_lineal, p=10, Tf=20
    )

    create_Table_Lambda(bests_lin, times_lin,bests_exp, times_exp,bests_log, times_log)

def SA_p(series, k_values):
    pi = int(input("P inicial: "))
    pf = int(input("P final: "))
    step = int(input("Incremento: "))

    bests, times,p = ejecutar_experimento_p(
    series, k_values, "p", enfriamiento_lineal,pf=pf,pi=pi,step=step)
    create_Table_p(bests, times,p)

def SA_L(series, k_values):
    Li = int(input("L inicial: "))
    Lf = int(input("L final: "))
    step = int(input("Incremento: "))

    bests, times,L = ejecutar_experimento_L(
    series, k_values, "L", enfriamiento_lineal,Lf=Lf,Li=Li,step=step)
    create_Table_L(bests, times,L)



def SA_T(series, k_values):
    T0 = float(input("T inicial: "))
    Tf = float(input("T final: "))
    p = float(input("Valor de p (enfriamiento lineal): "))

    bests, times = ejecutar_experimento(
        series, k_values, "T", enfriamiento_lineal, p=p, Tf=Tf, T0=T0)

    create_Table_T(bests, times, T0, Tf)


# ============================================================
# 9. GRÁFICAS (GUARDAR PNG)
# ============================================================


def plot_SA(mean_bests, mean_time, nombre):
    mean_bests = np.array(mean_bests)
    mean_time = np.array(mean_time)

    for TS in range(len(mean_bests[0])):
        plt.figure(figsize=(10, 5))
        plt.title(f"SA - Evolución RMSE TS{TS + 1} ({nombre})")

        rmses = [rmse[TS]["rmse"] for rmse in mean_bests]
        plt.plot(rmses, label="RMSE", color="blue", marker="o")

        plt.xlabel("time")
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




def create_Table_Lambda(bests_lin, times_lin,bests_exp, times_exp,bests_log, times_log):
    rows=[]
    bests=[]
    bests.append(bests_lin)
    bests.append(bests_exp)
    bests.append(bests_log)

    times=[]
    times.append(times_lin)
    times.append(times_exp)
    times.append(times_log)

    names=["Lineal","Exponencial","Logaritmico"]

    for N_val in range(len(names)):
        for TS in range(len(bests[0][0])):
            rmse = [bests[N_val][rep][TS]["rmse"] for rep in range(len(bests[0]))]
            time = [times[N_val][rep][TS] for rep in range(len(bests))]
            rows.append({
                "Enfriamiento": names[N_val],
                "Serie": f"Serie {TS + 1}",
                "RMSE Medio": np.mean(rmse),
                "RMSE Desv": np.std(rmse),
                "Tiempo Medio (s)": np.mean(time)
            })



    tabla = pd.DataFrame(rows)
    print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", showindex=False))



def create_Table_p(bests, times, p):
    rows=[]
    for p_val in range(len(p)):
        for TS in range(len(bests[0])):
            rmse = [bests[rep][TS][p_val]["rmse"] for rep in range(len(bests))]
            time = [times[rep][TS][p_val] for rep in range(len(bests))]
            rows.append({
                "p": p[p_val],
                "Serie": f"Serie {TS + 1}",
                "RMSE Medio": np.mean(rmse),
                "RMSE Desv": np.std(rmse),
                "Tiempo Medio (s)": np.mean(time)
            })


    
    tabla = pd.DataFrame(rows)
    print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", showindex=False))


def create_Table_L(bests, times, L):
    rows=[]
    for L_val in range(len(L)):
        for TS in range(len(bests[0])):
            rmse = [bests[rep][TS][L_val]["rmse"] for rep in range(len(bests))]
            time = [times[rep][TS][L_val] for rep in range(len(bests))]
            rows.append({
                "L": L[L_val],
                "Serie": f"Serie {TS + 1}",
                "RMSE Medio": np.mean(rmse),
                "RMSE Desv": np.std(rmse),
                "Tiempo Medio (s)": np.mean(time)
            })



    tabla = pd.DataFrame(rows)
    print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", showindex=False))

def create_Table_T(bests, times, T0,Tf):
    rows=[]
    for TS in range(len(bests[0])):
        rmse = [bests[rep][TS]["rmse"] for rep in range(len(bests))]
        time = [times[rep][TS] for rep in range(len(bests))]
        rows.append({
        "T0": T0,
        "Tf": Tf,
        "Serie": f"Serie {TS + 1}",
        "RMSE Medio": np.mean(rmse),
        "RMSE Desv": np.std(rmse),
        "Tiempo Medio (s)": np.mean(time)
        })
    tabla = pd.DataFrame(rows)
    print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", showindex=False))
