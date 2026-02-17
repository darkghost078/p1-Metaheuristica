import random

def puntuakAusazko(a,n):
    n2 = n - 1
    b = []
    while(n2>1):
        r = random.randint(1,len(a)-1)
        if r not in b:
            b.append(r)
            n2 = n2 - 1
    b.sort()
    print(b)

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n = int(input("Introduce el numero de puntos: "))

puntuakAusazko(a,n)