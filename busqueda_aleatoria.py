import time
from linear_regression import estimate_coef
from puntosRandom import puntuakAusazko
from RMSE import RMSE

def calcular_rmse_medio_segmentos(serie, puntos_corte):

    #Calcula la media de los RMSE de cada segmento individual.
    
    limites = [0] + puntos_corte + [len(serie)]
    rmse_total = 0.0
    n_segmentos = len(limites) - 1

    for i in range(n_segmentos):
        inicio = limites[i]
        fin = limites[i + 1]
        x_seg = list(range(inicio, fin))
        y_real = serie[inicio:fin]

        if len(y_real) < 2:
            continue  #Segmento de un solo punto: error 0

        b_0, b_1 = estimate_coef(x_seg, y_real)
        y_est = [b_0 * x + b_1 for x in x_seg]
        rmse_total += RMSE(y_real, y_est)

    return rmse_total / n_segmentos


def ejecutar_busqueda_aleatoria(serie, k, num_iteraciones=100):
    mejores_resultados = {'rmse': float('inf'), 'puntos': None}
    rmse_acumulado = 0
    tiempos = []
    detalles_iteracion = []  #Historial para que el main lo escriba en el CSV

    for iteracion in range(num_iteraciones):
        puntos_corte = puntuakAusazko(len(serie), k - 1)
        puntos_corte_original = puntos_corte.copy()

        inicio_tiempo = time.time()

        #media de RMSE por segmento
        rmse = calcular_rmse_medio_segmentos(serie, puntos_corte)

        fin_tiempo = time.time()
        tiempo = fin_tiempo - inicio_tiempo

        rmse_acumulado += rmse
        tiempos.append(tiempo)

        if rmse < mejores_resultados['rmse']:
            mejores_resultados['rmse'] = rmse
            mejores_resultados['puntos'] = puntos_corte_original

        detalles_iteracion.append({
            'iteracion': iteracion + 1,
            'puntos_corte': puntos_corte_original,
            'rmse': rmse,
            'tiempo': tiempo
        })

    rmse_promedio = rmse_acumulado / num_iteraciones
    tiempo_promedio = sum(tiempos) / len(tiempos)

    return mejores_resultados, rmse_promedio, tiempo_promedio, detalles_iteracion