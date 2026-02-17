import time
import csv
import os

from linear_regression import estimate_all_coef, estimate_all_points
from puntosRandom import puntuakAusazko 
from RMSE import RMSE
from lectura_datos import leer_serie

def main():
    archivos = ['TS1', 'TS2', 'TS3', 'TS4']

    series_temporales = []
    for archivo in archivos:
        serie = leer_serie(archivo)
        series_temporales.append(serie)
    
    k_por_serie = [9, 10, 20, 50]
    
    print("=" * 20)
    print("BÚSQUEDA ALEATORIA")
    print("=" * 20)
    

    #guardo los resultados
    nombre_archivo = 'resultados_busqueda_aleatoria.csv'
    mejores_globales = 'mejores_resultados_globales.csv' #archivo con los mejores resultados globales
    
    mejores_previos = {}
    if os.path.exists(mejores_globales):
        with open(mejores_globales, 'r') as f:
            reader = csv.reader(f)
            next(reader)  #salto cabecera
            for row in reader:
                serie = row[0]
                rmse = float(row[1])
                puntos = eval(row[2])
                mejores_previos[serie] = {'rmse': rmse, 'puntos': puntos}
    
    with open(nombre_archivo, 'w', newline='') as f_detalle:
        writer = csv.writer(f_detalle)
        writer.writerow(['serie', 'iteracion', 'puntos_corte', 'rmse_medio', 'tiempo'])
        
        #cada serie
        for idx_serie, (serie, k) in enumerate(zip(series_temporales, k_por_serie)):
            print(f"\n{'=' *20}") 
            print(f"    Procesando Serie {idx_serie+1} (k={k})")
            print(f"    Longitud: {len(serie)} puntos")
            print(f"{'='* 20}")
            
            mejores_resultados = {'rmse': float('inf'), 'puntos': None}
            rmse_acumulado = 0
            tiempos = []
            
            for iteracion in range(100):
                puntos_corte = puntuakAusazko(len(serie), k-1) #puntos de corte aleatorios
                puntos_corte_original = puntos_corte.copy()

                inicio_tiempo = time.time()

                coeficientes = estimate_all_coef(serie, puntos_corte.copy()) #regresiones
                puntos_estimados = estimate_all_points(coeficientes, puntos_corte.copy(), len(serie))
                rmse = RMSE(serie, puntos_estimados)
                
                fin_tiempo = time.time()
                tiempo = fin_tiempo - inicio_tiempo
                
                #estadísticas
                rmse_acumulado += rmse
                tiempos.append(tiempo)
                
                #guardo el mejor resultado
                if rmse < mejores_resultados['rmse']:
                    mejores_resultados['rmse'] = rmse
                    mejores_resultados['puntos'] = puntos_corte_original
                
                writer.writerow([
                    f'serie_{idx_serie+1}',
                    iteracion + 1,
                    str(puntos_corte_original),
                    f"{rmse:.6f}",
                    f"{tiempo:.6f}"
                ])
                
                
                if (iteracion + 1) %10 == 0:
                    print(f"  Iteración {iteracion+1:3d}/100 - Mejor RMSE: {mejores_resultados['rmse']:.6f}")
            
            #resultados finales de la serie 
            rmse_promedio = rmse_acumulado / 100
            tiempo_promedio = sum(tiempos) / len(tiempos)
            
            print(f"\n RESULTADOS SERIE {idx_serie+1}:")
            print(f" Mejor RMSE: {mejores_resultados['rmse']:.6f}")
            print(f" RMSE promedio: {rmse_promedio:.6f}")
            print(f" Mejor configuración: {mejores_resultados['puntos']}")
            print(f" Tiempo promedio por iteración: {tiempo_promedio:.6f} segundos")
    
            #comparo con el mejor global anterior
            nombre_serie = f'serie_{idx_serie+1}'
            if nombre_serie in mejores_previos:
                mejor_anterior = mejores_previos[nombre_serie]['rmse']
                if mejores_resultados['rmse'] < mejor_anterior:
                    print(f"!! Se ha mejorado el anterior: {mejor_anterior:.6f}")
                    mejores_previos[nombre_serie] = {'rmse': mejores_resultados['rmse'], 'puntos': mejores_resultados['puntos']}
                else:
                    print(f" El mejor global sigue siendo: {mejor_anterior:.6f} (ejecución anterior)")
            else:
                mejores_previos[nombre_serie] = {'rmse': mejores_resultados['rmse'], 'puntos': mejores_resultados['puntos']}
    
    #guardo los mejores globales actualizados
    with open(mejores_globales, 'w', newline='') as f_global:
        writer_global = csv.writer(f_global)
        writer_global.writerow(['serie', 'mejor_rmse', 'mejor_configuracion'])
        for serie, datos in mejores_previos.items():
            writer_global.writerow([serie, f"{datos['rmse']:.6f}", str(datos['puntos'])])


    print(f"\n{'=' * 30}")
    print(f"Los resultados se han guardado en: {nombre_archivo}")
    print(f"{'=' * 30}")

if __name__ == "__main__":
    main()