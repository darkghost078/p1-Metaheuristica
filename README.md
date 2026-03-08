# p1-Metaheuristica

## Acknowledgements

Este proyecto ha sido creado por:

-   000Volk000 -> [Darío Martínez Kostyuk](https://github.com/000Volk000) (p32makod@uco.es)
-   darkghost078 -> [David Martínez Molina](https://github.com/darkghost078) (i32marmd@uco.es)
-   i32mufea -> [Alicia Muriel Fernandez](https://github.com/i32mufea) (i32mufea@uco.es)
-   PMMS22 -> [Pablo Miguel Martin segovia](https://github.com/PMMS22) (i32masep@uco.es)
-   i32camol -> [Lucía Cañero Moslero](https://github.com/i32camol)(i32camol@uco.es)


# Instalacion

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

# Visualización del código

La forma más cómoda de explorar el proyecto es abrir los cuadernos Jupyter, donde está todo explicado y ejecutado paso a paso:
```bash
RS/RS.ipynb
HC/HC.ipynb
SA/SA.ipynb
```

Para abrirlos:
```bash
jupyter notebook 
```
o 
```bash
jupyter lab 
```

# Ejecución

Si prefieres ejecutar los algoritmos directamente sin usar Jupyter, cada carpeta contiene un main.py. 
Por ejemplo, para ejecutar Random Search:

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
cd HC
python3 main_SA.py
```

Los resultados y gráficas se generarán automáticamente en cada carpeta.
