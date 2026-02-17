import re

def leer_serie(nombre_archivo):
    try:
        with open(nombre_archivo, 'r') as f:
            contenido = f.read()
            
        #extraigo todos los números del contenido
        numeros = re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", contenido)
        if not numeros:
            numeros = re.findall(r"[-+]?\d*\.\d+", contenido)
        
        serie = [float(num) for num in numeros]
        
        print(f"Leídos {len(serie)} puntos de {nombre_archivo}")
        return serie
        
    except FileNotFoundError:
        print(f"ERROR! No se encuentra el archivo {nombre_archivo}")
        return []
    except Exception as e:
        print(f"ERROR! al leer {nombre_archivo}: {e}")
        return []