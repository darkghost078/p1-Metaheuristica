from simulated_annealing import (
    read_serie, ejecutar_experimento,
    enfriamiento_geometrico, enfriamiento_lineal, enfriamiento_logaritmico,
    plot_SA, plot_final_SA
)

import pandas as pd
from tabulate import tabulate


# ============================================================
# 1. CARGA DE SERIES
# ============================================================

files = ["../datos/TS1", "../datos/TS2", "../datos/TS3", "../datos/TS4"]
series = [read_serie(f) for f in files]
k_values = [9, 10, 20, 50]


# ============================================================
# 2. EJECUCIÓN DE LOS TRES MÉTODOS
# ============================================================

iters_geo, bests_geo, times_geo = ejecutar_experimento(
    series, k_values, "geométrico", enfriamiento_geometrico, p=0.95
)

iters_lin, bests_lin, times_lin = ejecutar_experimento(
    series, k_values, "lineal", enfriamiento_lineal, p=0
)

iters_log, bests_log, times_log = ejecutar_experimento(
    series, k_values, "logarítmico", enfriamiento_logaritmico, p=0
)


# ============================================================
# 3. GRÁFICAS
# ============================================================

plot_SA(bests_geo, iters_geo, series, "Geométrico")
plot_SA(bests_lin, iters_lin, series, "Lineal")
plot_SA(bests_log, iters_log, series, "Logarítmico")

plot_final_SA(series, bests_geo, k_values, "Geométrico")
plot_final_SA(series, bests_lin, k_values, "Lineal")
plot_final_SA(series, bests_log, k_values, "Logarítmico")


# ============================================================
# 4. MÉTRICAS
# ============================================================

def calcular_metricas(resultados, tiempos):
    rmse_medios = []
    rmse_desv = []
    tiempos_medios = []

    for TS in range(4):
        rmse_final = [rep[TS][-1]["rmse"] for rep in resultados]
        tiempo_total = [sum(rep[TS]) for rep in tiempos]

        rmse_medios.append(pd.Series(rmse_final).mean())
        rmse_desv.append(pd.Series(rmse_final).std())
        tiempos_medios.append(pd.Series(tiempo_total).mean())

    return rmse_medios, rmse_desv, tiempos_medios


rmse_geo, desv_geo, tiempo_geo = calcular_metricas(bests_geo, times_geo)
rmse_lin, desv_lin, tiempo_lin = calcular_metricas(bests_lin, times_lin)
rmse_log, desv_log, tiempo_log = calcular_metricas(bests_log, times_log)


tabla = pd.DataFrame({
    "Método": ["Geométrico", "Lineal", "Logarítmico"],
    "RMSE medio": [rmse_geo, rmse_lin, rmse_log],
    "Desviación típica": [desv_geo, desv_lin, desv_log],
    "Tiempo medio (s)": [tiempo_geo, tiempo_lin, tiempo_log]
})

tabla_formateada = tabla.copy()
tabla_formateada["RMSE medio"] = tabla_formateada["RMSE medio"].map(lambda x: f"{x:.6f}")
tabla_formateada["Desviación típica"] = tabla_formateada["Desviación típica"].map(lambda x: f"{x:.6f}")
tabla_formateada["Tiempo medio (s)"] = tabla_formateada["Tiempo medio (s)"].map(lambda x: f"{x:.6f}")

print(tabulate(tabla_formateada, headers="keys", tablefmt="fancy_grid", showindex=False))

