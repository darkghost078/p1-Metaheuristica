import sys
sys.path.append("..")

import math
import random
import time

from puntosRandom import randomPoints
from linear_regression import estimate_all_coef, estimate_all_points
from RS.busqueda_aleatoria import mean_rmse

enfriamiento_lineal = lambda T, T0, i, B: T0 - (i * B)
enfriamiento_geometrico = lambda T, T0, i, param: T * param
enfriamiento_boltzmann = lambda T, T0, i, param: T0 / (1 + math.log(i))


def random_neighbour(serie, n):
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

    return random.choice(p)

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
            srand = random_neighbour(sol, len(serie))

            rmse_srand = mean_rmse(serie, srand)

            delta = rmse_srand - rmse_sol

            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / max(T, 1e-8)):
                sol = srand

                rmse_sol = rmse_srand
    
                if rmse_sol < rmse_best:
                    best = sol.copy()
                    rmse_best = rmse_sol

            T = funcion_enfriamiento(T, T0, iter, p)
            iter+=1
    
    endTime = time.time()
    t = endTime - startTime
    return {'rmse': rmse_best, 'points': best}, t

# if __name__ == "__main__":
#     serie = [1,3,5,7,10,9,8,7]
#     alpha = lambda T, i: T-i
#     print(simulated_annealing(30,alpha, 100 ,10, serie,1))

if __name__ == "__main__":
    serie = [1,3,5,7,10,9,8,7]
    
    # Ejemplo usando enfriamiento Geométrico con alpha = 0.95
    mejor_solucion = simulated_annealing(
        T0=100, 
        funcion_enfriamiento=enfriamiento_boltzmann, 
        p=0.95,  # Este es el valor de alpha
        L=10, 
        Tf=0.1, 
        serie=serie, 
        k=2 # Queremos 2 segmentos, generará 1 punto de corte
    )
    
    print(mejor_solucion)