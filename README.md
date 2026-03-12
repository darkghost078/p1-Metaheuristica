# p1-Metaheuristica

## Acknowledgements

Este proyecto ha sido creado por:

-   000Volk000 -> [Darío Martínez Kostyuk](https://github.com/000Volk000) (p32makod@uco.es)
-   darkghost078 -> [David Martínez Molina](https://github.com/darkghost078) (i32marmd@uco.es)
-   i32mufea -> [Alicia Muriel Fernandez](https://github.com/i32mufea) (i32mufea@uco.es)
-   PMMS22 -> [Pablo Miguel Martin segovia](https://github.com/PMMS22) (i32masep@uco.es)
-   i32camol -> [Lucía Cañero Moslero](https://github.com/i32camol)(i32camol@uco.es)


# Instalación

## Entorno virtual

En la terminal ejecuta:

```bash
python -m venv .venv
```

Para activarlo ponemos:

```bash
source .venv/bin/activate
```

Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

# Ejecución

## Opción 1: Ejecutar cada algoritmo desde Python (modo script)

Cada carpeta contiene un main_*.py que es el main de cada metaheuristica.

Para Random Search:
```bash
cd RS
python3 main_RS.py
```

Para Hill Climbing:
```bash
cd HC
python3 main_HC.py
```

Para Simulated Annealing:
```bash
cd SA
python3 main_SA.py
```

Los resultados (tablas, métricas y gráficas) se generan automáticamente en cada carpeta o se muestran por terminal.


## Opción 2: Ejecutar los cuadernos de jupyter
Si prefieres ejecutarlo de forma visual, puedes usar los cuadernos Jupyter incluidos en las carpetas:

- `RS/`
- `HC/`
- `SA/`

Cada una contiene un cuaderno `.ipynb` con la implementación completa del algoritmo, sus gráficas y análisis paso a paso.

Además, en la carpeta principal encontrarás un cuaderno adicional:

- `Comparativa.ipynb`

Este cuaderno reúne los resultados de los tres algoritmos y muestra la comparación final de métricas y segmentaciones.

Para abrir los cuadernos:

```bash
jupyter notebook
