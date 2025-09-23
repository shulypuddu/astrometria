import numpy as np
# Acá voy a definir todas las funciones y distribuciones que voy a usar en el práctico 2

def t(x,chi,mu,sigma):
    """
    Argumento de la funcion distribución de Fisher-Tippet
    
    --Parámetros--



    --Retorna--
    
    
    """
    if chi !=0:
        return np.exp(-(x-mu)/sigma)
    else:
        return (1 + chi*(x-mu)/sigma)**(-1/chi)



def ac_Fisher_Tippet(x,chi,mu,sigma):
    """
    Función de distribución acumulativa de Fisher-Tippet

    --Parámetros--


    --Retorna--

    """
    

    return np.exp(-t(x,chi,mu,sigma))

def inv_Fisher_Tippet(u,chi,mu,sigma):
    """
    """
        
    if chi !=0:
        return mu + (sigma/chi)*( ( -np.log(u) )**(-chi) - 1 )
    else:
        return -np.log(np.log(u))*sigma + mu

def invCPDF(lambda_val, U):
    return -(1 / lambda_val) * np.log(1 - U)




def simular_buffon(N, l, t):

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

def estimar_varianza(datos):
    return np.var(datos, ddof=1)  # ddof=1 para obtener la varianza muestral (n-1)

def bootstrap_varianza(datos, num_resamples=1000, alpha=0.05):

    # Lista para almacenar las varianzas calculadas en cada remuestreo
    varianzas_bootstrap = []

    # Realizar num_resamples remuestreos
    for _ in range(num_resamples):
        # Crear una muestra con reemplazo de los datos
        muestra_bootstrap = np.random.choice(datos, size=len(datos), replace=True)
        # Calcular la varianza de la muestra
        varianza_muestra = estimar_varianza(muestra_bootstrap)
        varianzas_bootstrap.append(varianza_muestra)

    # Calcular percentiles para el intervalo de confianza
    limite_inferior = np.percentile(varianzas_bootstrap, (alpha / 2) * 100)
    limite_superior = np.percentile(varianzas_bootstrap, (1 - alpha / 2) * 100)

    return limite_inferior, limite_superior

