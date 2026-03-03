import math
import random
from puntosRandom import puntuakAusazko
from linear_regression import estimate_all_coef, estimate_all_points
from RMSE import RMSE

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

def simulated_annealing(T0, alpha, L, Tf, serie, k):
    T = T0
    sol = puntuakAusazko(len(serie), k)

    best = sol
    iter = 1
    while T >= Tf:
        for count in range(L):
            srand = random_neighbour(sol, len(serie))

            coeficientes = estimate_all_coef(serie, srand.copy())
            puntos_estimados = estimate_all_points(coeficientes, srand.copy(), len(serie))
            rmse_srand = RMSE(serie, puntos_estimados)

            coeficientes = estimate_all_coef(serie, sol.copy())
            puntos_estimados = estimate_all_points(coeficientes, sol.copy(), len(serie))
            rmse_sol = RMSE(serie, puntos_estimados)

            delta = rmse_srand - rmse_sol

            if random.uniform(0, 1) < math.exp(-delta / T) or delta < 0 :
                sol = srand

                coeficientes = estimate_all_coef(serie, best.copy())
                puntos_estimados = estimate_all_points(coeficientes, best.copy(), len(serie))
                rmse_best = RMSE(serie, puntos_estimados)
    
                if rmse_sol < rmse_best:
                    best = sol

            T = alpha(T, iter)
            iter+=1

    return best

if __name__ == "__main__":
    serie = [1,3,5,7,10,9,8,7]
    alpha = lambda T, i: T-i
    print(simulated_annealing(30,alpha, 100 ,10, serie,1))