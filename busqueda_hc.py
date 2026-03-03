import time
from linear_regression import estimate_all_coef, estimate_all_points
from hillClimbing import hc
from puntosRandom import puntuakAusazko 
from RMSE import RMSE

def busqueda_hc(serie,k):
    current = hc(serie, puntuakAusazko(len(serie),k))
    aux = hc(serie, current)

    coeficientes = estimate_all_coef(serie, current.copy())
    puntos_estimados = estimate_all_points(coeficientes, current.copy(), len(serie))
    crmse = RMSE(serie, puntos_estimados)

    coeficientes = estimate_all_coef(serie, aux.copy())
    puntos_estimados = estimate_all_points(coeficientes, aux.copy(), len(serie))
    armse = RMSE(serie, puntos_estimados)

    while(crmse > armse):
        current = aux
        crmse = armse

        aux = hc(serie, current)

        coeficientes = estimate_all_coef(serie, aux.copy())
        puntos_estimados = estimate_all_points(coeficientes, aux.copy(), len(serie))
        armse = RMSE(serie, puntos_estimados)

    return current

if __name__ == "__main__":
    serie = [1,3,5,7,10,9,8,7]
    print(busqueda_hc(serie,3))
