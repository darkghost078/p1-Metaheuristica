import sys
sys.path.append("..")

from lectura_datos import read_serie
from busqueda_aleatoria import randomSearch
import matplotlib.pyplot as plt
from linear_regression import estimate_all_points, estimate_all_coef

def main():
    start = int(input("Introduzca el numero de individuos iniciales: "))
    end = int(input("Introduzca el numero de individuos finales: "))
    increment = int(input("Introduzca el tamaño de los incrementos: "))
    
    if start<0 or end < start:
        print("Valores no validos")
        return 0
    
    files = ['../TS1', '../TS2', '../TS3', '../TS4']
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
        times=[]
        iters=[]
        bests=[]
        current_serie=series[TS]
        current_k=k_values[TS]
        for i in range(start,end,increment):
            if i==start:
                best, time, _= randomSearch(current_serie,current_k,increment)
            else:
                best, time, _= randomSearch(current_serie,current_k,increment,best)


            iters.append(i)
            bests.append(best.copy())
            times.append(time)

        series_iters.append(iters)
        series_bests.append(bests)
        series_times.append(times)


#Graficas
    #Grafica Iters vs RMSE
    for TS in range(len(series_times)):
        rmses = [element["rmse"] for element in series_bests[TS]]
        plt.plot(series_iters[TS],rmses,marker ='o')
        plt.title(f"RS {files[TS][3:]}")
        plt.xlabel("Tiempo")
        plt.ylabel("Mejor RMSE")
        plt.grid(True)
        plt.savefig(f"{files[TS][3:]}_ItersvsRMSE.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    #Grafica de puntos resultado
    for TS in range(len(series_times)):
        plt.title(f"RS {files[TS][3:]}")
        #Linea real
        plt.plot(series[TS], color='blue', linewidth=1, label="Datos reales")

        #Lineas verticales
        best=series_bests[TS][len(series_bests[TS])-1]['points']
        for i in range(len(best)):
            plt.axvline(x=best[i],linestyle='--',linewidth=1)

        #Linea estimada
        coef = estimate_all_coef(series[TS],best)
        plt.plot(estimate_all_points(coef,best,len(series[TS])), color='red', linewidth=1, label="Datos estimados")

        plt.legend()
        plt.grid(True)
        plt.savefig(f"{files[TS][3:]}_sol.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        


if __name__ == "__main__":
    main()
    print("No me he cagado encima")