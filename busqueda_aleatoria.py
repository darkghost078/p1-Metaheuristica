import time
from linear_regression import estimate_all_coef, estimate_all_points
from puntosRandom import puntuakAusazko 
from RMSE import RMSE

def ejecutar_busqueda_aleatoria(serie, k, num_iteraciones=100):
    mejores_resultados = {'rmse': float('inf'), 'puntos': None}
    rmse_acumulado = 0
    tiempos = []
    detalles_iteracion = [] # Guardamos el historial para que el main lo escriba en el CSV
    
    for iteracion in range(num_iteraciones):
        puntos_corte = puntuakAusazko(len(serie), k-1)
        puntos_corte_original = puntos_corte.copy()

        inicio_tiempo = time.time()

        coeficientes = estimate_all_coef(serie, puntos_corte.copy())
        puntos_estimados = estimate_all_points(coeficientes, puntos_corte.copy(), len(serie))
        rmse = RMSE(serie, puntos_estimados)
        
        fin_tiempo = time.time()
        tiempo = fin_tiempo - inicio_tiempo
        
        # Estadísticas
        rmse_acumulado += rmse
        tiempos.append(tiempo)
        
        # Guardar el mejor resultado
        if rmse < mejores_resultados['rmse']:
            mejores_resultados['rmse'] = rmse
            mejores_resultados['puntos'] = puntos_corte_original
            
        # Guardar detalle de esta vuelta
        detalles_iteracion.append({
            'iteracion': iteracion + 1,
            'puntos_corte': puntos_corte_original,
            'rmse': rmse,
            'tiempo': tiempo
        })
        
    rmse_promedio = rmse_acumulado / num_iteraciones
    tiempo_promedio = sum(tiempos) / len(tiempos)
    
    return mejores_resultados, rmse_promedio, tiempo_promedio, detalles_iteracion