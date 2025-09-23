#%%
import numpy as np
import matplotlib.pyplot as plt
import fun2 as f
import scipy
#Este primer bloque contiene las importaciones y la configuración inicial para todo el resto.
#CORRER PRIMERO ESTE BLOQUE SI O SI 


#%%
#---------------------------- EJERCICIO 3 ----------------------------#
x=np.random.uniform(0,1,1000)
y=np.random.uniform(0,1,1000)

def fisher_tippett_random(lambda_val, size=1000):
    U = np.random.uniform(0, 1, size)
    return - (1 / lambda_val) * np.log(-np.log(U))

# Parametros
lambda_val = 1.0
size = 10000

# Generamos las nuestras aleatorias
random_samples = fisher_tippett_random(lambda_val, size)

# Calcular la media
sample_mean = np.mean(random_samples)
expected_mean = 0.57721 / lambda_val

print(f"Media de la muestra: {sample_mean}")
print(f"Media teórica: {expected_mean}")

# Gráficos
plt.figure(figsize=(12, 6))

# Histograma
plt.subplot(1, 2, 1)
plt.hist(random_samples, bins=50, color='skyblue')
plt.title('Números random de Fisher-Tippett')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')

# CDF
sorted_samples = np.sort(random_samples)
cdf = np.arange(1, size+1) / size

plt.subplot(1, 2, 2)
plt.plot(sorted_samples, cdf, color='blue', lw=2)
plt.title('CDF)')
plt.xlabel('Valores')
plt.ylabel('CDF')

plt.tight_layout()
plt.show()
#%%
#---------------------------- EJERCICIO 4 ----------------------------#
# Parametros
lambda_val = 5  # eventos por hora

N = 17
np.random.seed(5)
U = np.random.random(size=N)
t = invCPDF(lambda_val, U)

t = np.append(0,t)  #le agrego el  tiempo cero
suma_eventos = np.cumsum(t)  #calculo el tiempo acumulado

plt.plot(np.arange(N+1), suma_eventos)
plt.title('Proceso Poisson: Distribución de tiempos')
plt.xlabel('Eventos')
plt.ylabel('Tiempo (horas)')
plt.grid(True)
plt.show()


#%% 
#---------------------------- EJERCICIO 8 ----------------------------#

# Parámetros
N = 100000  # Número de simulaciones
l = 29  # Longitud de la aguja
t = 30  # Separación entre las rayas (debe ser mayor que l)

# Realizar la simulación
pi_estimado = f.simular_buffon(N, l, t)

# Mostrar resultado
print(f"Estimación de pi después de {N} lanzamientos: {pi_estimado}")

# Comparar con el valor real de pi
print(f"Valor real de pi: {np.pi}")


#%%
#---------------------------- EJERCICIO 9 ----------------------------#

# Generar datos
np.random.seed(37)
datos = np.random.normal(loc=0, scale=1, size=100)

# Calcular la varianza original
varianza_original = f.estimar_varianza(datos)

# Calcular los intervalos de confianza con Bootstrap
lim_inf, lim_sup = f.bootstrap_varianza(datos, num_resamples=1000, alpha=0.05)

# Graficar el histograma de las varianzas obtenidas con Bootstrap
varianzas_bootstrap = [f.estimar_varianza(np.random.choice(datos, size=len(datos), replace=True)) for _ in range(1000)]


plt.hist(varianzas_bootstrap, bins=30, alpha=0.7, color='skyblue')
plt.axvline(varianza_original, color='red', linestyle='dashed', linewidth=2, label='Varianza Original')
plt.axvline(lim_inf, color='green', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
plt.axvline(lim_sup, color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Varianza')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de la varianza (Bootstrap) y su intervalo de confianza')
plt.show()


#%%
#---------------------------- EJERCICIO 10 ---------------------------#





