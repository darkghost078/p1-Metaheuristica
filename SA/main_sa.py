from simulated_annealing import (
    read_serie, ejecutar_experimento,
    enfriamiento_geometrico, enfriamiento_lineal, enfriamiento_logaritmico,
    plot_SA, plot_final_SA, SA_Lambda, SA_p, SA_L, SA_T
)

import pandas as pd


# ============================================================
# 1. CARGA DE SERIES
# ============================================================

files = ["../datos/TS1", "../datos/TS2", "../datos/TS3", "../datos/TS4"]
series = [read_serie(f) for f in files]
k_values = [9, 10, 20, 50]


print(f"1.Cambiar lambda\n2.Cambiar num iters\n3.Cambiar temperatura\n4.Cambiar p en el lineal")
opcion = input("Selecciona una opción: ")
if opcion == "1":
    SA_Lambda(series, k_values)
elif opcion == "2":
    SA_L(series,k_values)
elif opcion == "3":
    SA_T(series,k_values)
elif opcion == "4":
    SA_p(series, k_values)
elif opcion == "5":
    print("Saliendo del programa...")





