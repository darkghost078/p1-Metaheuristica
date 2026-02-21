import csv
import os
from lectura_datos import leer_serie
from busqueda_aleatoria import ejecutar_busqueda_aleatoria
# from hill_climbing import ejecutar_hill_climbing  <-- ¡Lo descomentaremos cuando lo hagamos!

def main():
    archivos = ['TS1', 'TS2', 'TS3', 'TS4']
    k_por_serie = [9, 10, 20, 50] # Valores de k exigidos en el enunciado[cite: 17, 29, 43, 58].
    
    series_temporales = []
    for archivo in archivos:
        serie = leer_serie(archivo)
        series_temporales.append(serie)
    
    print("=" * 20)
    print("EJECUCIÓN DE EXPERIMENTOS")
    print("=" * 20)

    archivo_ba = 'resultados_busqueda_aleatoria.csv'
    mejores_globales = 'mejores_resultados_globales.csv'
    
    # Cargar los mejores previos
    mejores_previos = {}
    if os.path.exists(mejores_globales):
        with open(mejores_globales, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                mejores_previos[row[0]] = {'rmse': float(row[1]), 'puntos': eval(row[2])}
    
    # Bucle principal para Búsqueda Aleatoria
    with open(archivo_ba, 'w', newline='') as f_detalle:
        writer = csv.writer(f_detalle)
        writer.writerow(['serie', 'iteracion', 'puntos_corte', 'rmse_medio', 'tiempo'])
        
        for idx_serie, (serie, k) in enumerate(zip(series_temporales, k_por_serie)):
            nombre_serie = f'serie_{idx_serie+1}'
            print(f"\n{'=' *20}\n    Procesando {nombre_serie} (k={k})\n    Longitud: {len(serie)} puntos\n{'='* 20}")
            
            # --- 1. EJECUTAR BÚSQUEDA ALEATORIA ---
            print(">> Ejecutando Búsqueda Aleatoria...")
            mejores_res, rmse_prom, tiempo_prom, detalles = ejecutar_busqueda_aleatoria(serie, k, num_iteraciones=100)
            
            # Escribir los resultados detallados en el CSV
            for det in detalles:
                writer.writerow([nombre_serie, det['iteracion'], str(det['puntos_corte']), f"{det['rmse']:.6f}", f"{det['tiempo']:.6f}"])
                if det['iteracion'] % 10 == 0:
                    print(f"  Iteración {det['iteracion']:3d}/100 - Mejor RMSE: {mejores_res['rmse']:.6f}")
            
            # Resumen de esta serie
            print(f"\n RESULTADOS {nombre_serie} (Búsqueda Aleatoria):")
            print(f" Mejor RMSE: {mejores_res['rmse']:.6f}")
            print(f" RMSE promedio: {rmse_prom:.6f}")
            print(f" Tiempo promedio por iteración: {tiempo_prom:.6f} s")
            
            # --- AQUÍ PODREMOS AÑADIR EL HILL CLIMBING LUEGO ---
            # mejores_res_hc, ... = ejecutar_hill_climbing(serie, k, ...)
            
            # Comparar y actualizar el mejor global
            if nombre_serie in mejores_previos:
                mejor_anterior = mejores_previos[nombre_serie]['rmse']
                if mejores_res['rmse'] < mejor_anterior:
                    print(f"!! Se ha mejorado el anterior global: {mejor_anterior:.6f}")
                    mejores_previos[nombre_serie] = {'rmse': mejores_res['rmse'], 'puntos': mejores_res['puntos']}
                else:
                    print(f" El mejor global sigue siendo: {mejor_anterior:.6f}")
            else:
                mejores_previos[nombre_serie] = {'rmse': mejores_res['rmse'], 'puntos': mejores_res['puntos']}
    
    # Guardar los mejores globales actualizados
    with open(mejores_globales, 'w', newline='') as f_global:
        writer_global = csv.writer(f_global)
        writer_global.writerow(['serie', 'mejor_rmse', 'mejor_configuracion'])
        for serie, datos in mejores_previos.items():
            writer_global.writerow([serie, f"{datos['rmse']:.6f}", str(datos['puntos'])])

    print(f"\n{'=' * 30}\nLos resultados se han guardado.\n{'=' * 30}")

if __name__ == "__main__":
    main()