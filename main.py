import csv
import os
from lectura_datos import leer_serie
from busqueda_aleatoria import ejecutar_busqueda_aleatoria

def main():
    start = int(input("Introduzca el numero de individuos iniciales: "))
    end = int(input("Introduzca el numero de individuos finales: "))
    increment = int(input("Introduzca el tamaño de los incrementos: "))
    
    if start<0 or end < start:
        print("Valores no validos")
        return 0
    
    files = ['TS1', 'TS2', 'TS3', 'TS4']
    k_values = [9, 10, 20, 50]
    
    series = []
    for file in files:
        serie = leer_serie(file)
        series.append(serie)
    
    print("\n" + "=" * 20)
    print("EJECUCIÓN DE EXPERIMENTOS")
    print("=" * 20)

    rmses=[]
    iters=[]

    for TS in len(series):
        current_serie=series[TS]
        current_k=k_values[TS]
        for i in range(start,end,increment):
            

if __name__ == "__main__":
    main()