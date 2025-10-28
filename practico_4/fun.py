import numpy as np
import scipy.stats as st

#---------------------------------------------------------------------------------
def datos_lineal(n):
    """
    Genero n datos tales que siguen una distribución lineal con un cierto ruido
    """
    a = 7
    b = 11
    eps = st.norm.rvs(loc=0, scale=2.5, size=n)  # ruido gaussiano
    x   = st.uniform.rvs(loc=3, scale=10, size=n)  # x uniforme
    y = a*x+b + eps

    return x,y


def modelo_lineal(x,parametros):
    a=parametros[0]
    b=parametros[1]
    return a*x+b


def log_prior_cte(parametros,cota_inf,cota_sup):
    """
    Quiero ver el logaritmo de la probabilidad de los parametros
    0 si estan dentro de las cotas, -inf sino
    """
    if np.all(parametros>cota_inf) and np.all(parametros<cota_sup):
        return 0 #devuelve 0 por que estoy pensando en un prior cte
    else:
        return -np.inf

def log_prior_normal(parametros,mu,sigma):
    """
    Quiero ver el logaritmo de la probabilidad de los parametros
    Utiliza st.norm.logpdf que calcula el logartimo de la pdf de una normal dados su mu y sigma
    Quiero que devuelva un array con los logaritmos de las probabilidades de cada parametro
    """
    p=[]
    for i in range(len(parametros)):
        p.append(st.norm.logpdf(parametros[i], loc=mu[i], scale=sigma))
    return p


def likelihood(datos,parametros,sigma,modelo):
    """
    Calculo el logaritmo del likelihood asumiendo errores gaussianos con desviacion estandar sigma
    --Parametros--
    datos: valor observado
    modelo: funcion que modela los datos
    parametros: parametros del modelo
    sigma: desviacion estandar de los errores
    """
    x_obs, y_obs = datos
    y_pred = modelo(x_obs, parametros)
    residuos = y_obs - y_pred
    return np.sum(st.norm.logpdf(residuos, loc=0, scale=sigma))


def fun_lum(x,phi,M,alfa):
    """
    Funcion de Schechter, es el modelo teorico que vamos a ajustar

    --Parametros--
    x: tipo array, valores a evaluar
    phi, M, alfa: tipo reales, parametros de la funcion
    """
    l= 0.4*np.log(10)*phi*10**(-0.4*(x-M)*(1+alfa))*np.e**(-10**(-0.4*(x-M)))
    return l


def log_post(modelo, datos,parametros):
    """
    Calculo el logaritmo de la probabilidad posterior
    """
    prior= log_prior_cte(parametros,cota_inf=[5,5],cota_sup=[15,15])
    like= likelihood(datos, parametros, sigma=0.5, modelo=modelo)
    return like + prior

def salto(parametros,cota_inf,cota_sup):
    """
    Genero un nuevo punto en el espacio de parametros
    """
    #nuevos_parametros= np.zeros(len(parametros))
    #for i in range(len(parametros)):
    #    nuevos_parametros[i]= st.norm.rvs(loc=parametros[i], scale=0.1)
    #return nuevos_parametros
    nuevos_parametros = parametros.copy()
    for i in range(len(parametros)):
        nuevos_parametros[i] += np.random.uniform(-1,1) *0.001* (cota_sup[i]-cota_inf[i])
    return nuevos_parametros # para el modelo lineal

def cadena_mcmc(modelo, datos, N, parametros_iniciales,cota_inf,cota_sup):
    """
    Genero una cadena de Markov Monte Carlo
    """
    cadena= np.zeros((N, len(parametros_iniciales))) #matriz
    cadena[0,:]= parametros_iniciales
    for i in range(1,N):
        parametros_actuales= cadena[i-1,:]
        nuevos_parametros= salto(parametros_actuales,cota_inf,cota_sup)
        p_actual= log_post(modelo, datos, parametros_actuales)
        p_nuevo= log_post(modelo, datos, nuevos_parametros)
        if np.isneginf(p_nuevo): # descarto los parametros si quedan fuera de las cotas
            cadena[i,:]= parametros_actuales 
        else:
            c= np.exp(p_nuevo-p_actual) # pues estoy calculando log de la probabilidad
            if c >= 1:
                cadena[i,:]= nuevos_parametros
            else: 
                r= np.random.uniform(0,1)
                if r < c:
                     cadena[i,:]= nuevos_parametros
                else:
                    cadena[i,:]= parametros_actuales
    return cadena[35]
