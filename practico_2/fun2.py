import numpy as np
from scipy.special import factorial
import math

# Acá voy a definir todas las funciones y distribuciones que voy a usar en el práctico 2


def Fisher_Tippet(x,xi,mu,sig):
    """
    Función de densidad de probabilidad de Fisher-Tippet

    --Parámetros--
    x: array - valor en el que se evalúa la función
    xi: float - parámetro de forma de la distribución
    mu: float - parámetro de localización de la distribución
    sigma: float - parámetro de escala de la distribución

    --Retorna--
    f: array - valor de la función de densidad en x
    """
    if xi == 0:
        t=(x - mu) / sig
        ft = (1/sig)*np.exp(-t-np.exp(-t))
    else:
        ft=(1/sig)*(1+xi*(x-mu)/sig)**(-1/xi - 1)*np.exp(-(1+xi*(x-mu)/sig)**(-1/xi))
    return ft
    


def ac_Fisher_Tippet(x,xi,mu,sig):
    """
    Función de distribución acumulativa de Fisher-Tippet

    --Parámetros--
    x: valor en el que se evalúa la función


    --Retorna--
    F: valor de la función de distribución acumulada en x
    """
    t=(x - mu) / sig
    if xi == 0:
        aft = np.exp(-np.exp(-t))
    else:
        aft = np.exp(-(1+xi*t)**(-1/xi))
    return aft

def inv_Fisher_Tippet(y,xi,mu,sig):
    """
    Función inversa de la distribución acumulada de Fisher-Tippet
    --Parámetros--
    u: valores de entrada (float entre 0 y 1 )
    chi: parámetro de forma de la distribución
    mu: parámetro de localización de la distribución
    sigma: parámetro de escala de la distribución

    --Retorna--
    x: valor correspondiente a la probabilidad acumulada u

    """
    if xi == 0:
        # Caso Gumbel: x = μ - σ * ln(-ln(F))
        return mu - sig * np.log(-np.log(y))
    else:
        # Caso general GEV: x = μ + (σ/ξ) * [(-ln(F))^(-ξ) - 1]
        return mu + (sig/xi) * ((-np.log(y))**(-xi) - 1)


def poisson(x,l):
    return (np.exp(-l) *l**x) / factorial(x)

def invCPDF(lambda_val, U):
    """
    Función inversa de la distribución acumulada de una distribución de Poisson continua(?)

    --Parámetros--
    lambda_val: tasa de eventos por unidad de tiempo
    U: valor(es) de entrada (float entre 0 y 1 )

    --Retorna--
    t: valor(es) correspondiente(s) al tiempo entre eventos
    """
    return -(1 / lambda_val) * np.log(1 - U)




def simular_buffon(N, l, t):
    """
    
    
    """
    cruces = 0  # Contador de cuántas veces la aguja toca una raya

    for _ in range(N):
        x = np.random.uniform(0, t/2)  # Distancia a la raya más cercana
        theta = np.random.uniform(0, np.pi/2)  # Ángulo agudo con las rayas

        # Condición para que la aguja cruce una raya
        if (l/2) * np.sin(theta) > x:
            cruces += 1

    # Estimar pi usando la fórmula de Buffon
    pi_est = (2 * l * N) / (t * cruces)
    return pi_est

def varianza(x):
    x0=np.mean(x)
    n=len(x)
    sum=0
    for i in range(n):
        sum+=(x0-x[i])**2
    return np.sqrt(sum/(n-1))

    
def bootstrap(x, func, m=1000):
    """
    --Parámetros--

    x=datos
    func=estadistico que quiero calcular
    m=cantidad de veces que se va a realizar el bootstrap

    --Retorna--
    y=lista de largo m con el valor de la funcion para cada muestra bootstrap

    """
    _y = np.zeros(m)
    for i in range(m):
        _x=np.random.choice(x,size=len(x),replace=True)
        _y[i]= func(_x)

    return _y 
def intervalos_bootstrap(y,alpha=0.05):
    """
    --Parámetros--
    y: lista de remuestreo bootstrap (hecho a partir de la funcion boootstrap)
    alpha: valor del intervalo de confianza (0.05 equivale a 95% de confianza)

    --Retorna--
    lim_inf y lim_sup valores extremos a partir de los cuales definimos las regiones de rechazo
    
    """

    lim_inf = np.percentile(y,(1-alpha/2)*100)
    lim_sup = np.percentile(y,(alpha/2)*100) 
    
    return lim_inf, lim_sup

def chi2(O,E):
    """
    --Parametros--
    Ambos objetos son listas y deben ser del mismo largo m
    O: de la muestra
    E: de la distribución teórica
        
    --Retorna--
    suma de las desviaciones cuadradas entre O y E normalizadas por E
    """
    chi2=0
    
    for i in range(len(O)):
        if E[i]!=0:
            chi2+=(O[i]-E[i])**2/E[i]
    return chi2

def binomial(n,p,k):
    """
    --Parámetros--

    --Retorna--
    
    """
    y=[]
    for i in range(1,k):

         b=np.log(math.factorial(n)) - (np.log(math.factorial(i)) + np.log(math.factorial(n-i))) + i*np.log(p) + (n-i)*np.log(1-p)
         f= np.exp(b)
         y.append(f)
    return y
