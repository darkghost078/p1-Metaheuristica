import sys
sys.path.append("..")

from lectura_datos import read_serie
from simulated_annealing import simulated_annealing, enfriamiento_lineal, enfriamiento_geometrico, enfriamiento_boltzmann
import matplotlib.pyplot as plt
from linear_regression import estimate_all_points, estimate_all_coef
import numpy as np


def mean(serie):
    means = []
    for TS in range(len(serie[0])):
        mean_s=[]
        for i in range(len(serie)):
            for j in range(len(serie[0][TS])):
                if i == 0:
                    mean_s.append(serie[i][TS][j]['rmse'])
                else:
                    mean_s[j] += serie[i][TS][j]['rmse']
        for i in range(len(mean_s)):
            mean_s[i] /= len(serie)
        means.append(mean_s)
    
    return means

def std(serie,means):
    stds = []
    for TS in range(len(serie[0])):
        std_s=[]
        for i in range(len(serie)):
            for j in range(len(serie[0][TS])):
                diff=serie[i][TS][j]['rmse']-means[TS][j]
                diff**=2
                if i == 0:
                    std_s.append(diff)
                else:
                    std_s[j] += diff

        for i in range(len(std_s)):
            std_s[i]=pow(std_s[i],2)
            std_s[i] /= len(serie)
            std_s[i]=np.sqrt(std_s[i])

        stds.append(std_s)
    
    return stds

def main():
    print("\n--- SELECCIÓN DE ENFRIAMIENTO ---")
    print("1. Lineal")
    print("2. Geométrico / Exponencial")
    print("3. Boltzmann (Logarítmico)")
    opcion = input("Elija una opción (1/2/3): ")

    if opcion == '1':
        funcion_enf = enfriamiento_lineal
        param_enf = 0.1 # Valor de Beta (ajústalo si baja muy rápido)
        nombre_enf = "Lineal"
    elif opcion == '2':
        funcion_enf = enfriamiento_geometrico
        param_enf = 0.95 # Valor de Alpha
        nombre_enf = "Geometrico"
    elif opcion == '3':
        funcion_enf = enfriamiento_boltzmann
        param_enf = 1.0 # En tu función de boltzmann este parámetro no afecta mucho, pero hay que pasarlo
        nombre_enf = "Boltzmann"
    else:
        print("Opción no válida. Usando Geométrico por defecto.")
        funcion_enf = enfriamiento_geometrico
        param_enf = 0.95
        nombre_enf = "Geometrico"

    start = int(input("Introduzca el valor inicial de L: "))
    end = int(input("Introduzca el valor final de L: "))
    increment = int(input("Introduzca el tamaño de los incrementos de L: "))
    
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
    print("EJECUCIÓN DE EXPERIMENTOS: SIMULATED ANNEALING")
    print("=" * 20)

    T0 = 100
    Tf = 0.1
    p_alpha = 0.95

    #Listas para guardar iters y best de cada serie

    mean_iters=[]
    mean_bests=[]
    mean_times=[]
    for _ in range (20):
        series_iters=[]
        series_bests=[]
        series_times=[]
        for TS in range(len(series)):
            times=[]
            iters=[]
            bests=[]
            current_serie=series[TS]
            current_k=k_values[TS]
            for L in range(start,end+1,increment):
                best, time = simulated_annealing(
                    T0=T0, 
                    funcion_enfriamiento=funcion_enf, 
                    p=param_enf, 
                    L=L, 
                    Tf=Tf, 
                    serie=current_serie, 
                    k=current_k
                )

                iters.append(L)
                bests.append(best.copy())
                times.append(time)

            series_iters.append(iters)
            series_bests.append(bests)
            series_times.append(times)
        
        mean_times.append(series_times)
        mean_bests.append(series_bests)
        mean_iters.append(series_iters)

#Graficas

    mean_series=mean(mean_bests)
    std_series=std(mean_bests,mean_series)


    std_mean_neg=[]
    std_mean_pos=[]
    for TS in range(len (series_times)):
        std_neg=[]
        std_pos=[]
        for i in range(len(mean_series[TS])):
            std_pos.append(mean_series[TS][i]+std_series[TS][i])
            std_neg.append(mean_series[TS][i]-std_series[TS][i])
        std_mean_neg.append(std_neg)
        std_mean_pos.append(std_pos)

    for TS in range(len(series_times)):
        plt.figure()
        plt.title(f"SA ({nombre_enf}) Mean RMSE {files[TS][3:]}")
        plt.plot(series_iters[TS], mean_series[TS], color='blue', linewidth=1, label="Media RMSE")
        plt.plot(series_iters[TS], std_mean_neg[TS], color='red', linewidth=1, linestyle='--', label="Desv inf")
        plt.plot(series_iters[TS], std_mean_pos[TS], color='red', linewidth=1, linestyle='--', label="Desv sup")
        plt.xlabel("Valor de L")
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig(f"SA_{nombre_enf}_{files[TS][3:]}_mean.png", dpi=300, bbox_inches='tight')
        plt.close()

    #Grafica de puntos resultado
    for TS in range(len(series_times)):
        plt.figure()
        plt.title(f"SA ({nombre_enf}) {files[TS][3:]}") 
        plt.plot(series[TS], color='blue', linewidth=1, label="Datos reales")
        best=series_bests[TS][len(series_bests[TS])-1]['points']
        for i in range(len(best)):
            plt.axvline(x=best[i],linestyle='--',linewidth=1)
        coef = estimate_all_coef(series[TS],best)
        plt.plot(estimate_all_points(coef,best,len(series[TS])), color='red', linewidth=1, label="Datos estimados")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"SA_{nombre_enf}_{files[TS][3:]}_sol.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
    print("No me he cagado encima")