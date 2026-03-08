from busqueda_aleatoria import (
    read_serie, ejecutar_RS, plot_RS, plot_final_solutions,
    exactitud_RS, variabilidad_RS, tiempo_RS
)

import pandas as pd
from tabulate import tabulate


# ============================================================
# 1. CARGA DE SERIES
# ============================================================

files = [
    "../datos/TS1",
    "../datos/TS2",
    "../datos/TS3",
    "../datos/TS4"
]

series = [read_serie(f) for f in files]
k_values = [9, 10, 20, 50]


# ============================================================
# 2. EJECUCIÓN DEL EXPERIMENTO
# ============================================================

resultados_RS = ejecutar_RS(series, k_values,
                            start=10, end=200, step=10,
                            repeticiones=20)


# ============================================================
# 3. GRÁFICAS
# ============================================================

plot_RS(series, resultados_RS)
plot_final_solutions(series, resultados_RS, k_values)


# ============================================================
# 4. MÉTRICAS
# ============================================================

rmse_medios = exactitud_RS(resultados_RS)
rmse_desv = variabilidad_RS(resultados_RS)
tiempos_medios = tiempo_RS(resultados_RS)

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
