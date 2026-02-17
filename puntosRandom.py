import random

def puntuakAusazko(a,n):
    b = []
    while(n>1):
        r = random.randint(1,len(a)-1)
        if r not in b:
            b.append(r)
            n = n - 1
    b.sort()
    print(b)

a = [0, 73748659823454631, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n = int(input("Introduce el numero de puntos: "))

puntuakAusazko(a,n)