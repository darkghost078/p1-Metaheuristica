from linear_regression import estimate_all_coef, estimate_all_points
from RMSE import RMSE
from puntosRandom import puntuakAusazko 

def search_neighbour(serie, n):
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

    return p

def hc(serie, k):
    solucion = puntuakAusazko(len(serie),k)
    print(solucion)
    coeficientes = estimate_all_coef(serie, solucion.copy())
    puntos_estimados = estimate_all_points(coeficientes, solucion.copy(), len(serie))
    rmse = RMSE(serie, puntos_estimados)
    flag = True

    while(flag):
        flag = False
        v = search_neighbour(solucion, len(serie))

        for vect in v:
            coeficientes = estimate_all_coef(serie, vect.copy())
            puntos_estimados = estimate_all_points(coeficientes, vect.copy(), len(serie))
            current = RMSE(serie, puntos_estimados)

            if current < rmse:
                solucion = vect
                rmse = current
                flag = True
        print(rmse)

    return solucion


if __name__ == "__main__":
    serie = [1,3,5,7,10,9,8,7]
    hc(serie,1)
