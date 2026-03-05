import sys
sys.path.append("..")

import time
from RS.busqueda_aleatoria import mean_rmse 
from puntosRandom import randomPoints 

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
    startTime = time.time()
    solucion = randomPoints(len(serie),k)
    rmse = mean_rmse(serie, solucion)
    flag = True

    while(flag):
        flag = False
        v = search_neighbour(solucion, len(serie))

        for vect in v:
            current = mean_rmse(serie, vect)

            if current < rmse:
                solucion = vect
                rmse = current
                flag = True
                break

    endTime = time.time()
    t = endTime - startTime
    
    # Devolvemos el formato estándar
    return {'rmse': rmse, 'points': solucion}, t


if __name__ == "__main__":
    serie = [1,3,5,7,100,90,80,70]
    print(hc(serie,1))
