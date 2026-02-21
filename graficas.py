import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_graficas_analisis(archivo_csv):
    # Leer los datos del CSV generado por el main
    df = pd.read_csv(archivo_csv)
    
    # Crear una figura con 3 subgráficas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Análisis Inicial: Búsqueda Aleatoria', fontsize=16)

    # 1. Gráfica de Exactitud (Mejor RMSE por serie)
    mejores_rmse = df.groupby('serie')['rmse_medio'].min().reset_index()
    sns.barplot(x='serie', y='rmse_medio', data=mejores_rmse, ax=axes[0], palette='viridis')
    axes[0].set_title('Exactitud: Mejor RMSE Encontrado')
    axes[0].set_ylabel('RMSE (Menos es mejor)')
    axes[0].set_xlabel('Serie Temporal')

    # 2. Gráfica de Variabilidad (Distribución de errores)
    sns.boxplot(x='serie', y='rmse_medio', data=df, ax=axes[1], palette='Set2')
    axes[1].set_title('Variabilidad: Distribución del Error')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xlabel('Serie Temporal')

    # 3. Gráfica de Tiempo (Promedio por serie)
    tiempos_medios = df.groupby('serie')['tiempo'].mean().reset_index()
    sns.barplot(x='serie', y='tiempo', data=tiempos_medios, ax=axes[2], palette='magma')
    axes[2].set_title('Eficiencia: Tiempo Medio por Iteración')
    axes[2].set_ylabel('Segundos')
    axes[2].set_xlabel('Serie Temporal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la imagen para el informe
    plt.savefig('analisis_busqueda_aleatoria.png')
    print("Gráfica guardada como 'analisis_busqueda_aleatoria.png'")

if __name__ == "__main__":
    generar_graficas_analisis('resultados_busqueda_aleatoria.csv')