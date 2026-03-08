from hill_climbing import (
    read_serie, ejecutar_HC, plot_HC_RMSE, plot_final_HC
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
# 2. EJECUCIÓN DEL EXPERIMENTO
# ============================================================

resultados_HC = ejecutar_HC(series, k_values, repeticiones=20)


# ============================================================
# 3. GRÁFICAS
# ============================================================

plot_HC_RMSE(series, resultados_HC)
plot_final_HC(series, resultados_HC, k_values)


# ============================================================
# 4. MÉTRICAS
# ============================================================

rmse_medios = [pd.Series([rep["rmse"] for rep in resultados_HC[i]]).mean() for i in range(4)]
rmse_desv = [pd.Series([rep["rmse"] for rep in resultados_HC[i]]).std() for i in range(4)]
tiempos_medios = [pd.Series([rep["tiempo"] for rep in resultados_HC[i]]).mean() for i in range(4)]

tabla = pd.DataFrame({
    "Serie": ["TS1 (k=9)", "TS2 (k=10)", "TS3 (k=20)", "TS4 (k=50)"],
    "RMSE medio": rmse_medios,
    "Desviación típica": rmse_desv,
    "Tiempo medio (s)": tiempos_medios
})

tabla_formateada = tabla.copy()
tabla_formateada["RMSE medio"] = tabla_formateada["RMSE medio"].map("{:.6f}".format)
tabla_formateada["Desviación típica"] = tabla_formateada["Desviación típica"].map("{:.6f}".format)
tabla_formateada["Tiempo medio (s)"] = tabla_formateada["Tiempo medio (s)"].map("{:.6f}".format)

print(tabulate(tabla_formateada, headers="keys", tablefmt="fancy_grid", showindex=False))

