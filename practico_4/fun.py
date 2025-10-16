import numpy as np
import scipy.stats as st

#---------------------------------------------------------------------------------
def datos_lineal(n):
    """
    Genero n datos tales que siguen una distribución lineal con un cierto ruido
    """
    a = 7
    b = 11
    eps = st.norm.rvs(loc=0, scale=2.78, size=n)  # ruido gaussiano
    x = st.uniform.rvs(loc=3, scale=10, size=n)  # x uniforme
    y = a*x+b + eps
    return x,y


def priors(parametros):
    
    return 0

def likelihood(modelo, datos,parametros):
    return 0

def posterior(modelo, datos,parametros):
    return 0

def salto():
    return 0








