#%%
#LIBRERIAS USADAS EN EL PROGRAMA
import numpy as np
import matplotlib.pyplot as plt
import fun2 as f
import scipy.stats as st
import math 
#Este primer bloque contiene las importaciones y la configuración inicial para todo el resto.
#CORRER PRIMERO ESTE BLOQUE SI O SI 
bar_color='lightsteelblue'
linea_color='tomato'

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
plt.hist(random_samples, bins='auto', color=bar_color, density=True )
plt.plot(sorted_samples,f.Fisher_Tippet(sorted_samples, xi=xi, mu=mu, sig=sig), '--',color=linea_color, label='PDF Teórica')
plt.title('Números random de Fisher-Tippett')
plt.xlim(-3.5, 12.75)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_random.jpg',dpi=300,bbox_inches='tight')
# Funcion acumulada empírica
plt.subplot(1, 2, 2)
plt.plot(sorted_samples, cdf, color='dodgerblue', lw=2)
plt.plot(sorted_samples, f.ac_Fisher_Tippet(sorted_samples, xi=xi, mu=mu, sig=sig ),'--',color=linea_color, label='CDF Teórica')
plt.title('CDF')
plt.xlim(-3.5, 12.75)
plt.xlabel('Valores')
plt.ylabel('CDF')
plt.tight_layout()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_analysis.jpg', dpi=300, bbox_inches='tight')
plt.show()

#%%
#---------------------------- EJERCICIO 4 ----------------------------#
# Parametros
lamda = 5  # eventos por hora
tiempo=3 #cantidad de horas
lambda_val= lamda*tiempo #el equivalente al problema para la dist de Poisson
N = 30
num_simulaciones = 100
xp=np.arange(N+1)
clientes_en_3h=[]
np.random.seed(252)
cant_eventos = np.zeros(30)
for i in range(30):
    U = np.random.random(size=N)
    t = f.invCPDF(lamda, U)
    t = np.append(0,t)  #le agrego el  tiempo cero
    suma_eventos = np.cumsum(t)  #calculo el tiempo acumulado
    plt.plot(xp,suma_eventos)
    eventos_antes_3h = np.sum(suma_eventos <= tiempo)
    clientes_en_3h.append(eventos_antes_3h)
plt.axhline(y=3, color=linea_color, linestyle='--')
plt.title('Proceso Poisson: Distribución de eventos en el tiempo')
plt.xlabel('Cantidad de clientes')
plt.ylabel('Tiempo (horas)')
plt.grid(True)
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/poisson.jpg',dpi=300,bbox_inches='tight')
plt.show()

x_teorico = np.arange(0, max(clientes_en_3h)+1)
prob_teorica = st.poisson.pmf(x_teorico, lambda_val)
freq_teorica = prob_teorica 

plt.plot(x_teorico, freq_teorica, 'ro-', color=linea_color, 
         linewidth=2, markersize=6, label=f'Poisson teórica (λ={lambda_val})')
plt.hist(clientes_en_3h, bins=np.arange(0, max(clientes_en_3h)+2) - 0.5,density=True,color=bar_color,label='Poisson empirica',edgecolor='black')

plt.xlabel('Número de clientes en 3 horas')
plt.ylabel('Frecuencia')
plt.title(f'Distribución de clientes en {tiempo} horas\n({num_simulaciones} simulaciones)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/poisson_completo.jpg', dpi=300, bbox_inches='tight')
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

plt.hist(pi_buffon,bins='auto',color=bar_color,density=True)
plt.axvline(x=np.pi, color=linea_color, linestyle='--', label=r'$\pi$')
plt.xlabel(r'Valor de $\pi$')
plt.title(f'Dispersion del experimento del Buffon con {N} tiradas')
plt.legend()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/buffon_x10000.jpg',dpi=300,bbox_inches='tight')
plt.show()
# Comparar con el valor real de pi
print(f"Valor real de pi: {np.pi}")
print(f"Valor obtenido de pi: {np.mean(pi_buffon)}")
print(f'Varianza de la muestra:{np.var(pi_buffon)} ')
x=np.arange(1,10000,50)
pi_2 =np.zeros(len(x))
for i in range(len(x)):
    pi_2[i]= f.simular_buffon(x[i],l,t)
plt.scatter(x,pi_2,color='yellowgreen')
plt.axhline(y=np.pi, color=linea_color, linestyle='--', label=r'$\pi$ verdadero')
plt.ylabel(r'Valor de $\pi$')
plt.xlabel('Cantidad de tiradas')
plt.legend()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/buffon_conv.jpg',dpi=300,bbox_inches='tight')
plt.show()
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

# Calcular la media original
media_og = np.mean(datos)

# Calcular los intervalos de confianza con Bootstrap
media_bootstrap = f.bootstrap(datos,np.mean)
m_lim_inf, m_lim_sup = f.intervalos_bootstrap(media_bootstrap)

# Graficar el histograma de las varianzas obtenidas con Bootstrap
fig, axs = plt.subplots(2, 1, figsize=(14, 18)) # 2 filas, 1 columna

axs[0].hist(varianzas_bootstrap, bins=30, alpha=0.7, color='lightsteelblue')
axs[0].axvline(varianza_original, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
axs[0].axvline(1, color='brown', linestyle='dashed', linewidth=2, label='Varianza Teórica')
axs[0].axvline(lim_inf, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
axs[0].axvline(lim_sup, color='tomato', linestyle='dashed', linewidth=2)
axs[0].set_xlabel('Varianza')
axs[0].set_ylabel('Frecuencia')
axs[0].legend()
axs[0].set_title('Distribución de la varianza (Bootstrap) y su intervalo de confianza')

axs[1].hist(media_bootstrap, bins=30, alpha=0.7, color='lightsteelblue')
axs[1].axvline(media_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Media Original')
axs[1].axvline(0, color='brown', linestyle='dashed', linewidth=2, label='Media Teórica')
axs[1].axvline(m_lim_inf, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
axs[1].axvline(m_lim_sup, color='tomato', linestyle='dashed', linewidth=2)
axs[1].set_xlabel('Media')
axs[1].set_ylabel('Frecuencia')
axs[1].legend()
axs[1].set_title('Distribución de la media (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/bootstrap.jpg',dpi=300,bbox_inches='tight')
plt.show()
#---- Ahora repito el remuestreo pero siguiendo una distribucion de Fisher-Tippett
#Valores originales
ft_x = f.inv_Fisher_Tippet(y, xi=0, mu=0, sig=1)
ft_mean_og = np.mean(ft_x)
ft_mean_teo = 0.57721
ft_var_og = f.varianza(ft_x)
ft_var_teo = np.sqrt(np.pi**2/6)
#Valores con bootstrap
ft_media_bs = f.bootstrap(ft_x,np.mean)
ft_var_bs = f.bootstrap(ft_x,f.varianza)
ft_lim_sup_var, ft_lim_inf_var = f.intervalos_bootstrap(ft_var_bs)
ft_lim_sup_media, ft_lim_inf_media = f.intervalos_bootstrap(ft_media_bs)

fig, ax = plt.subplots(2, 1, figsize=(14, 18)) # 2 filas, 1 columna


ax[0].hist(ft_media_bs, bins=30, alpha=0.7, color='lightsteelblue')
ax[0].axvline(ft_mean_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Media Original')
ax[0].axvline(ft_mean_teo, color='brown', linestyle='dashed', linewidth=2, label='Media Teórica')
ax[0].axvline(ft_lim_inf_media, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
ax[0].axvline(ft_lim_sup_media, color='tomato', linestyle='dashed', linewidth=2)
ax[0].set_xlabel('Media')
ax[0].set_ylabel('Frecuencia')
ax[0].legend()
ax[0].set_title('Distribución de la media (Bootstrap) y su intervalo de confianza')
ax[1].hist(ft_var_bs, bins=30, alpha=0.7, color='lightsteelblue')
ax[1].axvline(ft_var_teo, color='brown', linestyle='dashed', linewidth=2, label='Media Teórica')
ax[1].axvline(ft_var_og, color='yellowgreen', linestyle='dashed', linewidth=2, label='Varianza Original')
ax[1].axvline(ft_lim_inf_var, color='tomato', linestyle='dashed', linewidth=2, label=f'IC {100 * (1 - 0.05):.1f}% ')
ax[1].axvline(ft_lim_sup_var, color='tomato', linestyle='dashed', linewidth=2)
ax[1].set_xlabel('Varianza')
ax[1].set_ylabel('Frecuencia')
ax[1].legend()
ax[1].set_title('Distribución de la varianza (Bootstrap) y su intervalo de confianza')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/ft_bootstrap.jpg',dpi=300,bbox_inches='tight')
plt.show()
#%%
#---------------------------- EJERCICIO 10 ---------------------------#
cant=100 #cantidad de experimentos
n =10
p=0.4
x=np.arange(11)
teorica_dens=st.binom.pmf(x, n=n, p=p)
teorica=teorica_dens*100

plt.figure(figsize=(10, 6))
plt.bar(x, teorica_dens, alpha=0.7, color=bar_color, edgecolor='black')
plt.xlabel('Número de éxitos (k)')
plt.ylabel('Probabilidad')
plt.title(f'Distribución Binomial (n={n}, p={p})')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()


lista=np.zeros(100)
for i in range(100):        #veces que sorteo la variable
    _x=st.binom.rvs(n=10,p=0.4) #da variables aleatorias siguiendo la distribucion binomial
    lista[i]=_x         #las agrego en una lista

hist_result= plt.hist(lista,  bins=np.arange(-0.5, 11.5, 1),density=True ,color=linea_color)
plt.xlabel('Número de éxitos (k)')
plt.ylabel('Probabilidad')
plt.title(f'Distribución Binomial (n={n}, p={p})')
plt.bar(x, teorica_dens, alpha=0.7, color=bar_color, edgecolor='black')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/bin_chi2.jpg',dpi=300,bbox_inches='tight')
plt.show()

frecuencias_observadas = hist_result[0]  # Las frecuencias están en el primer elemento
print(f'Las frecuencias observadas son: {frecuencias_observadas}')
chi= f.chi2(frecuencias_observadas,teorica_dens) 
print(chi)

print('El valor-p de la prueba es:', f.pvalue(chi,10))

#%%
x_range=np.arange(-4,15,1)
teorica_norm_dens = st.binom.pmf(k=x_range, n=10, p=0.4)
teorica_norm = 100*teorica_norm_dens
frec_norm=[]
plt.figure(figsize=(17,7))

for i in [1,2,3,4,5,6]:  #posición en el gráfico
    plt.subplot(2,3,i)   #figura con 2 filas y 3 columnas
    
    #Grafico la binomial teorica
    plt.bar(x_range, teorica_norm, color=bar_color, alpha=0.8, label='Binomial')
    
    #Grafico las distintas normales, quiero mu=2,3,4,5,6,7
    h=plt.hist(f.empirica_normal(i+1), bins=np.arange(-4.5, 15.5, 1), histtype='step', ec=linea_color,linewidth=2 ,label='Normal: $\mu$ = '+ str(i+1))
    plt.legend(loc='best')
    frec_norm.append(h[0]) #agrego las frecuencias a una lista
plt.tight_layout()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/binomial_vs_normal.jpg', dpi=300, bbox_inches='tight')
plt.show()

#para cada una calculo chi2
chi_norm=[]
grados_libertad=(len(x_range)-1)
chi_critico=st.chi2.ppf((1-0.05),grados_libertad)
p_critico=f.pvalue(chi_critico,grados_libertad)
for i in range(len(frec_norm)):
    chi_norm.append(f.chi2(frec_norm[i],teorica_norm))
j=[2,3,4,5,6,7]
plt.scatter(j,chi_norm,color='yellowgreen')
plt.axhline(y=chi_critico, color=linea_color, linestyle='--', label=r'$\xi$ critico')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/dist_chi.jpg', dpi=300, bbox_inches='tight')
plt.show()
p_valor=[]
for i in range(len(j)):
    p_valor.append(f.pvalue(chi_norm[i],grados_libertad))
plt.scatter(j,p_valor,color='yellowgreen')
plt.axhline(y=p_critico, color=linea_color, linestyle='--', label=r'$\xi$ critico')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/dist_p.jpg', dpi=300, bbox_inches='tight')
plt.show()

#%%
x_range=np.arange(-4,15,1)
teorica_norm_dens = st.binom.pmf(k=x_range, n=10, p=0.4)
teorica_norm = 100*teorica_norm_dens
frec_norm=[]
plt.figure(figsize=(17,7))

for i in [1,2,3,4,5,6]:  #posición en el gráfico
    plt.subplot(2,3,i)   #figura con 2 filas y 3 columnas
    
    #Grafico la binomial teorica
    plt.bar(x_range, teorica_norm, color=bar_color, alpha=0.8, label='Binomial')
    
    #Grafico las distintas normales, quiero mu=2,3,4,5,6,7
    h=plt.hist(f.empirica_normal(i+1), bins=np.arange(-4.5, 15.5, 1), histtype='step', ec=linea_color,linewidth=2 ,label='Normal: $\mu$ = '+ str(i+1))
    plt.legend(loc='best')
    frec_norm.append(h[0]) #agrego las frecuencias a una lista
plt.tight_layout()
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/binomial_vs_normal.jpg', dpi=300, bbox_inches='tight')
plt.show()

#para cada una calculo chi2
chi_norm=[]
grados_libertad=(len(x_range)-1)
chi_critico=st.chi2.ppf((1-0.05),grados_libertad)
p_critico=f.pvalue(chi_critico,grados_libertad)
for i in range(len(frec_norm)):
    chi_norm.append(f.chi2(frec_norm[i],teorica_norm))
j=[2,3,4,5,6,7]
plt.scatter(j,chi_norm,color='yellowgreen')
plt.axhline(y=chi_critico, color=linea_color, linestyle='--', label=r'$\xi$ critico')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/dist_chi.jpg', dpi=300, bbox_inches='tight')
plt.show()
p_valor=[]
for i in range(len(j)):
    p_valor.append(f.pvalue(chi_norm[i],grados_libertad))
plt.scatter(j,p_valor,color='yellowgreen')
plt.axhline(y=p_critico, color=linea_color, linestyle='--', label=r'$\xi$ critico')
plt.savefig('/mnt/sda2/astrometria/practico_2/imagenes/dist_p.jpg', dpi=300, bbox_inches='tight')
plt.show()
