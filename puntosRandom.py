import random

def puntuakAusazko(longitud_serie, n_cortes):
    b = []
    while n_cortes > 0:
        r = random.randint(1, longitud_serie - 1)
        if r not in b:
            b.append(r)
            n_cortes = n_cortes - 1
            
    b.sort()
    return b