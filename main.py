from lectura_datos import read_serie
from busqueda_aleatoria import randomSearch
import matplotlib.pyplot as plt

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
        serie = read_serie(file)
        series.append(serie)
    
    print("\n" + "=" * 20)
    print("EJECUCIÓN DE EXPERIMENTOS")
    print("=" * 20)


    #Listas para guardar iters y best de cada serie
    series_iters=[]
    series_bests=[]
    series_times=[]
    
    for TS in range(len(series)):
        iters=[]
        bests=[]
        current_serie=series[TS]
        current_k=k_values[TS]
        for i in range(start,end,increment):
            best, times, rmses= randomSearch(current_serie,current_k,i)
            #Grafica
            iters.append(i)
            bests.append(best)

        series_iters.append(iters)
        series_bests.append(bests)
        series_times.append(times)




if __name__ == "__main__":
    main()
    print("No me he cagado encima")