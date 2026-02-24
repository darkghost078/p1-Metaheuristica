import pandas as pd
import matplotlib.pyplot as plt
import ast
from lectura_datos import leer_serie
from linear_regression import estimate_all_coef, estimate_all_points

def graficar_mejores_series(archivo_csv):
    # 1. Cargar el CSV de resultados
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"Error: No se encuentra {archivo_csv}")
        return
        
    # 2. Encontrar la fila con el mejor (menor) RMSE para cada serie
    mejores_resultados = df.loc[df.groupby('serie')['rmse_medio'].idxmin()]
    
    # Mapeo de los nombres del CSV a los archivos reales
    archivos_ts = {
        'serie_1': 'TS1', 
        'serie_2': 'TS2', 
        'serie_3': 'TS3', 
        'serie_4': 'TS4'
    }
    
    for index, row in mejores_resultados.iterrows():
        nombre_serie = row['serie']
        mejor_rmse = row['rmse_medio']
        
        # ast.literal_eval convierte el string "[10, 20, 30]" en una lista real de Python
        puntos_corte = ast.literal_eval(row['puntos_corte'])
        
        # 3. Leer los datos reales de la serie
        nombre_archivo = archivos_ts[nombre_serie]
        try:
            serie_real = leer_serie(nombre_archivo)
        except Exception as e:
            print(f"No se pudo leer la {nombre_archivo}: {e}")
            continue
            
        # 4. Generar los puntos de la recta usando vuestras funciones
        coeficientes = estimate_all_coef(serie_real, puntos_corte.copy())
        puntos_estimados = estimate_all_points(coeficientes, puntos_corte.copy(), len(serie_real))
        
        # 5. Crear la gráfica
        plt.figure(figsize=(14, 6))
        
        # Dibujar la serie original (Gris)
        plt.plot(serie_real, label='Serie Original', color='lightgray', linewidth=2)
        
        # Dibujar la aproximación de la regresión (Rojo)
        plt.plot(puntos_estimados, label=f'Aproximación (RMSE: {mejor_rmse:.4f})', color='red', linewidth=2.5)
        
        # Dibujar líneas verticales en los puntos de corte (Azul discontinuo)
        for pt in puntos_corte:
            plt.axvline(x=pt, color='blue', linestyle='--', alpha=0.5)
            
        plt.title(f'Segmentación de {nombre_serie.upper()} (k={len(puntos_corte)+1})', fontsize=16)
        plt.xlabel('Índice / Tiempo', fontsize=12)
        plt.ylabel('Valor', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 6. Guardar gráfica
        nombre_imagen = f'segmentacion_{nombre_serie}.png'
        plt.savefig(nombre_imagen)
        plt.close()
        print(f"¡Gráfica generada con éxito!: {nombre_imagen}")

if __name__ == "__main__":
    graficar_mejores_series('resultados_busqueda_aleatoria.csv')