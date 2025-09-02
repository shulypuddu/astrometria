import numpy as np

def glc(n,x0, a=57,c=1,m=256):
    """
    Linear Congruential Generator
     Usando un valor incial genera números aleatorios a partir de operaciones con la congruencia lineal 


    --Parámetros--
    Todos los parámetros iniciales son ENTEROS, igual q los núumeros aleatorios
    m es el módulo, nos dice el valor maximo que puede tomar el número aleatorio (m-1)
    a es el multiplicador
    c es el incremento
    x0 es la semilla o valor inicial
    
    --Retorna--
    Una lista de n números aleatorios entre 0 y m-1 

    """
    x = []

    for i in range(n):
        y = (c+a*x0)%m
        x.append(y)
        x0=y
    
    return x

def glc_int(n,x0, a=57,c=1,m=256):
    """
    Linear Congruential Generator
     Usando la funcion glc genera números aleatorios entre 0 y 1 a partir de operaciones con la congruencia lineal


    --Parámetros--
    Todos los parámetros iniciales son ENTEROS, igual q los núumeros aleatorios
    m es el módulo, nos dice el valor maximo que puede tomar el número aleatorio (m-1)
    a es el multiplicador
    c es el incremento
    x0 es la semilla o valor inicial
    
    --Retorna--
    Una lista de n números aleatorios entre 0 y 1 

    """
    return np.array(glc(n,x0,a,c,m))/m


def fib(n,x0, j=24, k=55, m=2**32):
    """
    Generador de Fibonacci con retardo
     Usando un valor incial genera números aleatorios a partir de operaciones con la congruencia lineal y a partir de k numeros aleatorios previamente definidos

    --Parámetros--
    todos los parámetros deben son números enteros 
    



    --Retorna--
    una lista de n numeros aleatorios 

    """

    numeros = glc(k,x0,a=1664525,c=1013904223,m=2**32)
    for i in range(k,k+n): #empieza desde k y va hasta k+n-1
        numeros.append(((numeros[i-j]+numeros[i-k])%m)/m)

    return numeros[k:]

def fib_int(n,x0, j=24, k=55, m=2**32):
    """
    Generador de Fibonacci con retardo
     Usando la funcion fib genera números aleatorios a partir de operaciones con la congruencia lineal y a partir de k numeros aleatorios previamente definidos

    --Parámetros--
    todos los parámetros deben son números enteros 
    



    --Retorna--
    una lista de n numeros aleatorios 

    """

    return np.array(fib(x0,n,j,k,m))/m
