#%%
#LIBRERIAS USADAS EN EL PROGRAMA
import numpy as np
import matplotlib.pyplot as plt
import fun2 as f
import scipy.stats as st
import math 
#Este primer bloque contiene las importaciones y la configuración inicial para todo el resto.
#CORRER PRIMERO ESTE BLOQUE SI O SI 


#%%
#---------------------------- EJERCICIO 3 ----------------------------#
n=1000
y=np.random.uniform(0,1,n)
xi=0
mu=0
sig=2
# Generamos las nuestras aleatorias
random_samples = f.inv_Fisher_Tippet(y, xi=xi, mu=mu, sig=sig)

sorted_samples = np.sort(random_samples)
cdf = np.arange(1, n + 1) / n


# Calcular la media
sample_mean = random_samples.mean()
expected_mean = 0.57721 *sig

print(f"Media de la muestra: {sample_mean}")
print(f"Media teórica: {expected_mean}")

# Histograma de las muestras
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(random_samples, bins='auto', color='lightsteelblue', density=True )
plt.plot(sorted_samples,f.Fisher_Tippet(sorted_samples, xi=xi, mu=mu, sig=sig), '--',color='tomato', label='PDF Teórica')
plt.title('Números random de Fisher-Tippett')
plt.xlim(-3.5, 12.75)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_random.png',dpi=300,bbox_inches='tight')
# Funcion acumulada empírica
plt.subplot(1, 2, 2)
plt.plot(sorted_samples, cdf, color='dodgerblue', lw=2)
plt.plot(sorted_samples, f.ac_Fisher_Tippet(sorted_samples, xi=xi, mu=mu, sig=sig), 'r--', label='CDF Teórica')
plt.title('CDF')
plt.xlim(-3.5, 12.75)
plt.xlabel('Valores')
plt.ylabel('CDF')
plt.tight_layout()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
#---------------------------- EJERCICIO 4 ----------------------------#
# Parametros
lamda = 5  # eventos por hora
tiempo=3 #cantidad de horas
lambda_val= lamda*tiempo #el equivalente al problema para la dist de Poisson
N = 18
xp=np.arange(N+1)

np.random.seed(5)
cant_eventos = np.zeros(18)
for i in range(18):
    U = np.random.random(size=N)
    t = f.invCPDF(lamda, U)
    t = np.append(0,t)  #le agrego el  tiempo cero
    suma_eventos = np.cumsum(t)  #calculo el tiempo acumulado
    plt.plot(xp,suma_eventos)
plt.title('Proceso Poisson: Distribución de eventos en el tiempo')
plt.xlabel('Cantidad de clientes')
plt.ylabel('Tiempo (horas)')
plt.grid(True)
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/poisson.png',dpi=300,bbox_inches='tight')
plt.show()

#poisson3h = suma_eventos[i](3)

#%% 
#---------------------------- EJERCICIO 8 ----------------------------#

# Parámetros
N = 10000  # Cantidad de tiradas
l = 50  # Longitud de la aguja
t = 200  # Separación entre las rayas (debe ser mayor que l)

# Realizar la simulación
pi_buffon = np.zeros(100)
for i in range(100): #repito 100 veces el experimento 
    pi_buffon[i] = f.simular_buffon(N, l, t)

plt.hist(pi_buffon,bins='auto',density=True)
plt.axvline(x=np.pi, color='red', linestyle='--', label='$\pi$')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/buffon_x100.png',dpi=300,bbox_inches='tight')

# Comparar con el valor real de pi
print(f"Valor real de pi: {np.pi}")


#%%
#---------------------------- EJERCICIO 9 ----------------------------#

# Generar datos
np.random.seed(37)
datos = np.random.normal(loc=0, scale=1, size=1000)

# Calcular la varianza original
varianza_original = f.varianza(datos)

# Calcular los intervalos de confianza con Bootstrap
varianzas_bootstrap = f.bootstrap(datos,f.varianza)
lim_inf, lim_sup = f.intervalos_bootstrap(varianzas_bootstrap)

# Graficar el histograma de las varianzas obtenidas con Bootstrap
plt.hist(varianzas_bootstrap, bins=30, alpha=0.7, color='lightsteelblue')
plt.axvline(varianza_original, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
plt.axvline(lim_inf, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
plt.axvline(lim_sup, color='tomato', linestyle='dashed', linewidth=2)
plt.xlabel('Varianza')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de la varianza (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/bootstrap_var.png',dpi=300,bbox_inches='tight')
plt.show()
# Calcular la media original
media_og = np.mean(datos)

# Calcular los intervalos de confianza con Bootstrap
media_bootstrap = f.bootstrap(datos,np.mean)
m_lim_inf, m_lim_sup = f.intervalos_bootstrap(media_bootstrap)

# Graficar el histograma de las varianzas obtenidas con Bootstrap
plt.hist(media_bootstrap, bins=30, alpha=0.7, color='lightsteelblue')
plt.axvline(media_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
plt.axvline(m_lim_inf, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
plt.axvline(m_lim_sup, color='tomato', linestyle='dashed', linewidth=2)
plt.xlabel('Media')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de la media (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/bootstrap_mean.png',dpi=300,bbox_inches='tight')
plt.show()
#---- Ahora repito el remuestreo pero siguiendo una distribucion de Fisher-Tippett
#Valores originales
ft_mean_og = np.mean(random_samples)
ft_var_og = f.varianza(random_samples)

#Valores con bootstrap
ft_media_bs = f.bootstrap(random_samples,np.mean)
ft_var_bs = f.bootstrap(random_samples,f.varianza)
ft_lim_sup_var, ft_lim_inf_var = f.intervalos_bootstrap(ft_var_bs)
ft_lim_sup_media, ft_lim_inf_media = f.intervalos_bootstrap(ft_media_bs)

plt.hist(ft_media_bs, bins=30, alpha=0.7, color='lightsteelblue')
plt.axvline(ft_mean_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
plt.axvline(ft_lim_inf_media, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
plt.axvline(ft_lim_sup_media, color='tomato', linestyle='dashed', linewidth=2)
plt.xlabel('Media')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de la media (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_bootstrap_mean.png',dpi=300,bbox_inches='tight')
plt.show()

plt.hist(ft_var_bs, bins=30, alpha=0.7, color='lightsteelblue')
plt.axvline(ft_var_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
plt.axvline(ft_lim_inf_var, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
plt.axvline(ft_lim_sup_var, color='tomato', linestyle='dashed', linewidth=2)
plt.xlabel('Varianza')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de la varianza (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_bootstrap_var.png',dpi=300,bbox_inches='tight')
plt.show()
#%%
#---------------------------- EJERCICIO 10 ---------------------------#
cant=100 #cantidad de experimentos
m=100 #cantidad de puntos a cada experimento
chi=np.zeros(m)
n =10
p=0.4
x=np.arange(m)
teo_bi_rel=st.binom.pmf(x, n=n, p=p)
teorica=teo_bi_rel*100
for i in range(cant):
    y= np.random.binomial(n,p, size=cant)
    chi[i]=f.chi2(y,teo_bi_rel)


plt.hist(y,bins=np.arange(-0.5, 11.5, 1),color='indianred',alpha=0.75)
plt.bar(x,teorica,color='lightsteelblue',alpha=0.75)
plt.xlim(-1,10)
plt.show()

plt.hist(chi,bins='auto')
