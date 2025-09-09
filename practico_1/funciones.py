
import numpy as np

def glc_int(n,x0, a=57,c=1,m=256):
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

def glc(n,x0, a=57,c=1,m=256):
    """
    Generador Lineal Congruencial:
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
    return np.array(glc_int(n,x0,a,c,m))/m


def fib_int(n, j=24, k=55, m=2**32):
    """
    Generador de Fibonacci con retardo
     Usando un valor incial genera números aleatorios a partir de operaciones con la congruencia lineal y a partir de k numeros aleatorios previamente definidos

    --Parámetros--
    todos los parámetros deben son números enteros 
    n: cantidad de valores
    x0: semilla inicial
    j,k: parametros de retraso ¡Importante!: k > j
    m : modulo 

    --Retorna--
    una lista de n numeros aleatorios 

    """
    x0= np.random.random()
    numeros =glc_int(k,x0,a=1664525,c=1013904223,m=2**32)
    
    for i in range(k,k+n): #empieza desde k y va hasta k+n-1
       nro=(numeros[i-j]+numeros[i-k])
       numeros.append(nro%m)
    
    return numeros[k:]

def fib(n, j=24, k=55, m=2**32):
    """
    Generador de Fibonacci con retardo
     Usando la funcion fib genera números aleatorios a partir de operaciones con la congruencia lineal y a partir de k numeros aleatorios previamente definidos

    --Parámetros--
    todos los parámetros deben son números enteros  
    n: cantidad de valores
    x0: semilla inicial
    j,k: parametros de retraso ¡Importante!: k>j
    m : modulo 

    --Retorna--
    una lista de n numeros aleatorios 

    """
    fib=[]
    for i in range(n):
        fibfib=fib_int(n,j,k,m)[i]/m
        fib.append(fibfib)

    return fib




def periodo(lista):
    """
    Calcula el periodo de una lista de numeros aleatorios

    --Parámetros--
    lista: lista de numeros aleatorios

    --Retorna--
    periodo: periodo de la lista (valor al partir del cual se repiten los números)
    """

    for i in range(1,len(lista)):
        if lista[i] == lista[0]:
            return i
    return 0 # en caso q no haya repeticiones devuelve 0



    
def momentok(k, lista):
    """
    Calcula el momento k-esimo de una lista de números

    --Parámetros--
    k= entero positivo, indica el momento a calcular
    lista= lista de números (pueden ser enteros o flotantes)
    
    --Retorna--
    Momento k-esimo (numero real) 
    si k=1 hablamos del valor de expectancia


    """
    _x= np.array(lista)
    _xk= _x**k    
    return np.sum(_xk)/len(_xk)





def pearson_correlation (x , y ) :
    """
    Calcula el coeficiente de correlacion de Pearson entre dos arrays

    Parameters :
    x, y: arrays de igual longitud

    Returns :
    r: coeficiente de correlacion de Pearson
    """
    # Verificar que tienen la misma longitud
    if len( x ) != len( y ) :
        raise ValueError ("Los arrays deben tener la misma longitud ")

    n=len(x)

    # Calcular medias
    mean_x = np . mean ( x )
    mean_y = np . mean ( y )

    # Calcular numerador y denominador
    numerator = np .sum (( x - mean_x ) * ( y - mean_y ) )
    denominator = np . sqrt ( np .sum (( x - mean_x ) **2) * np .sum (( y - mean_y ) **2) )

    # Evitar division por cero
    if denominator == 0:
        return 0
    return numerator / denominator


def dados(N,x0):
    """
    --Parámetros--
    N : cantidad de tiradas
    
    --Retorna--
    lista de N valores entre 0 y 6
    """
    nros=[]
    m=2**32	
    a=1664525
    c=1013904223
    _nros= glc(N,x0,a,c,m)
    for i in range(len(_nros)):
        nros.append(int(_nros[i]*6+1))

    return nros

def dado_doble(N):
    """
    --Parámetros--
    N : cantidad de tiradas
    n0 : semilla inicial
    
    --Retorna--
    lista de N valores entre 0 y 6
    """
    m=2**32	
    a=1664525
    c=1013904223
    nros=[]
    _nros= glc(N,1235598,a,c,m)
    for i in range(len(_nros)):
        nros.append(int(_nros[i]*10+1)+2)
    return nros


