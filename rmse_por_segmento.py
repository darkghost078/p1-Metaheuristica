import pandas as pd
import ast
import csv
from lectura_datos import leer_serie
from linear_regression import estimate_coef
from RMSE import RMSE

def guardar_rmse_segmentos_serie3():
    # 1. Leer el CSV para buscar la mejor solución de la serie_3
    try:
        df = pd.read_csv('resultados_busqueda_aleatoria.csv')
    except FileNotFoundError:
        print("Error: No se encuentra el archivo de resultados.")
        return

    # Filtramos por serie_3 y sacamos la fila con el RMSE mínimo
    datos_serie = df[df['serie'] == 'serie_3']
    
    if datos_serie.empty:
        print("Error: No hay datos para la serie_3 en el archivo CSV.")
        return
        
    mejor_fila = datos_serie.loc[datos_serie['rmse_medio'].idxmin()]
    puntos_corte = ast.literal_eval(mejor_fila['puntos_corte'])
    
    # 2. Leer los datos originales de TS3
    serie = leer_serie('TS3')
    
    # 3. Construir los límites de todos los segmentos
    # Añadimos el inicio (0) y el final (longitud total de la serie)
    limites = [0] + puntos_corte + [len(serie)]
    
    resultados_segmentos = []
    print(f"Analizando los {len(limites)-1} segmentos de la Serie 3 (TS3)...")
    print("-" * 50)
    rmse_acc=0
    
    # 4. Recorrer cada segmento y calcular su RMSE individual
    for i in range(len(limites) - 1):
        inicio = limites[i]
        fin = limites[i+1]
        
        # Extraer los valores reales (X e Y) solo de este trozo
        x_segmento = list(range(inicio, fin))
        y_real = serie[inicio:fin]
        
        # Calcular la recta de regresión matemática para este segmento
        b_0, b_1 = estimate_coef(x_segmento, y_real)
        
        # Generar los puntos Y estimados con esa recta
        y_estimado = [b_0 * x + b_1 for x in x_segmento]
        
        # Calcular el RMSE exclusivo de este segmento
        rmse_seg = RMSE(y_real, y_estimado)

        rmse_acc+=rmse_seg
        print()
        # Guardar los datos
        resultados_segmentos.append([f"Segmento {i+1}", inicio, fin, b_0, b_1, rmse_seg])
        print(f"Segmento {i+1:2d} (Puntos {inicio:4d} a {fin:4d}) | RMSE: {rmse_seg:.4f}")
    
    rmse_medio=rmse_acc/(len(limites) - 1)
    # 5. Escribir todo en un archivo CSV nuevo
    nombre_archivo = 'rmse_segmentos_serie3.csv'
    with open(nombre_archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Segmento', 'Punto_Inicio', 'Punto_Fin', 'Pendiente(b0)', 'Interseccion(b1)', 'RMSE'])
        writer.writerows(resultados_segmentos)
        
    print("-" * 50)
    print(f"¡Listo! El desglose por segmentos se ha guardado en: {nombre_archivo}")
    print(f"Media RMSE: {rmse_medio}")
    print(f"ACC RMSE: {rmse_acc}")

if __name__ == "__main__":
    guardar_rmse_segmentos_serie3()