

def glc(m, a, c,x0):
    
    """
    Linear Congruential Generator
    M es el módulo, nos dice el valor maximo que puede tomar el número aleatorio (M-1)
    a es el multiplicador
    c es el incremento
    x0 es la semilla o valor inicial
    
    """
    x = (c+a*x0)%m
    lista_aleatorios = []
    return x



