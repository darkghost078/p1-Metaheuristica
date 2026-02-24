import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_analisis_estadistico(df):
    """Genera la comparativa de Exactitud, Variabilidad y Tiempo."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Análisis Estadístico: Búsqueda Aleatoria', fontsize=16)

    # 1. Exactitud (Mejor RMSE por serie)
    mejores_rmse = df.groupby('serie')['rmse_medio'].min().reset_index()
    sns.barplot(x='serie', y='rmse_medio', data=mejores_rmse, ax=axes[0], palette='viridis')
    axes[0].set_title('Exactitud: Mejor RMSE Encontrado')
    axes[0].set_ylabel('RMSE (Menos es mejor)')
    axes[0].set_xlabel('Serie Temporal')

    # 2. Variabilidad (Distribución de errores)
    sns.boxplot(x='serie', y='rmse_medio', data=df, ax=axes[1], palette='Set2')
    axes[1].set_title('Variabilidad: Distribución del Error')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xlabel('Serie Temporal')

    # 3. Eficiencia (Tiempo Medio por Iteración)
    tiempos_medios = df.groupby('serie')['tiempo'].mean().reset_index()
    sns.barplot(x='serie', y='tiempo', data=tiempos_medios, ax=axes[2], palette='magma')
    axes[2].set_title('Eficiencia: Tiempo Medio por Iteración')
    axes[2].set_ylabel('Segundos')
    axes[2].set_xlabel('Serie Temporal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('analisis_estadistico_ba.png')
    print("Imagen 'analisis_estadistico_ba.png' generada.")

def generar_grafica_convergencia(df):
    plt.figure(figsize=(10, 6))

    for serie in df['serie'].unique():
        datos_serie = df[df['serie'] == serie].sort_values('iteracion')
        mejores_hasta_ahora = datos_serie['rmse_medio'].cummin()
        plt.plot(datos_serie['iteracion'], mejores_hasta_ahora, label=f' {serie}', linewidth=2)

    plt.title('Curva de Convergencia: Evolución del Mejor RMSE', fontsize=14)
    plt.xlabel('Número de Iteraciones', fontsize=12)
    plt.ylabel('RMSE (Menor es mejor)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('convergencia_ba.png')
    print("Imagen 'convergencia_ba.png' generada.")

if __name__ == "__main__":
    archivo = 'resultados_busqueda_aleatoria.csv'
    try:
        data = pd.read_csv(archivo)
        generar_analisis_estadistico(data)
        generar_grafica_convergencia(data)
    except FileNotFoundError:
        print(f"Error: No se encuentra {archivo}. Ejecuta primero el main.py")
