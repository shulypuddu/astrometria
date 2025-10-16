#%% ------------------------------------------------------------------
#---------- LIBRERIAS A USAR Y CARGA DE DATOS --------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy.optimize as opt


sns.set_palette("Accent") 
# Cargo los archivos y descarto las galaxias con blueshift
df0 = pd.read_csv('/mnt/sda2/astrometria/practico_3/P3_shulypuddu.csv')
df = df0[(df0['z'] > 0) & ((df0['u']-df0['r']) > 0.25) & ((df0['u']-df0['r']) < 4.5)]

# Separo en rojas y azules
rojo = df[(df['u']-df['r'])>2.5]
azul = df[(df['u']-df['r'])<2.5]
color_r = '#cd5c5c' #indianred
color_a = '#4169e1' #royalblue

# Separo en elipticas y espirales usando ZOOVOTES
elip = df[df['elip']>0.8]
esp = df[df['esp']>0.8]
mer = df[df['meg']>0.8]

# Separo en bulge y disco usando fracDeV_r
bulge = df[df['fracDeV_r']>0.8]
disco = df[df['fracDeV_r']<0.2]

ur = (df['u']-df['r']).dropna()
gr = (df['g']-df['r']).dropna()
ug = (df['u']-df['g']).dropna()

#%% ------------------------------------------------------------------
#---------- AJUSTE LINEAL CASERO -------------------------------------
def cuad_min(x, y):
    """
    --Parámetros--
    x, y : pandas Series o arrays 
    --Retorna--
    Los coeficientes a y b de la recta y = ax + b
    """
    _x = x.values  # Convertir a numpy array
    _y = y.values 
    
    # Calcular medias
    x_medio = np.mean(_x)
    y_medio = np.mean(_y)
    
    # Calcular diferencias (como arrays, no listas)
    delta_x = _x - x_medio
    delta_y = _y - y_medio
    
    # Fórmulas de mínimos cuadrados
    a = np.sum(delta_x * delta_y) / np.sum(delta_x**2)
    b = y_medio - a * x_medio
    
    return a, b

# Calculo la recta
muestra = df.sample(frac=0.05, random_state=252)
a, b = cuad_min(muestra['g'], muestra['r'])
y = a * muestra['g'] + b

#---------- AJUSTE DOBLE GAUSSIANA -------------------------------------

def doble_gaussiana(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    return (amp1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            amp2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

def gauss(x,amp,mu,sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


cuentas, bordes = np.histogram(ur, bins=1000,density=True)
centros = (bordes[:-1] + bordes[1:]) / 2

# Estimaciones iniciales para los parámetros
p0 = [1, ur.mean()-0.5, ur.std()/2, 1,ur.mean()+0.5, ur.std()/2]

# Ajuste
params, cov = opt.curve_fit(doble_gaussiana, centros, cuentas, p0=p0)

x_fit = np.linspace(ur.min(), ur.max(), 200)
x1_fit = np.linspace(ur.min(), 3, 200)
x2_fit = np.linspace( 1,ur.max(), 200)
g_azul= gauss(x1_fit,params[0],params[1],params[2])
g_rojo= gauss(x2_fit,params[3],params[4],params[5])

def diferencia(x):
    return gauss(x,params[0],params[1],params[2]) - gauss(x,params[3],params[4],params[5])
x_inicial= (params[4]+params[1])/2
x_cruce = opt.fsolve(diferencia, x_inicial)[0] 

#%% ------------------------------------------------------------------
#---------- PRIMEROS GRÁFICOS ----------------------------------------

sns.histplot((df['u']-df['r']),bins='auto',alpha=0.75,stat='density')
plt.plot(x_fit, doble_gaussiana(x_fit, *params), label='Ajuste Doble Gaussiana', color='darkslategray')
plt.plot(x1_fit, g_azul,  label='Gaussiana para azules',color=color_a) 
plt.plot(x2_fit, g_rojo, color=color_r, label='Gaussiana para rojas') 
plt.xlim(0.25,4.5)
plt.xlabel('u-r')
plt.savefig('/mnt/sda2/astrometria/practico_3/informe/imagenes/ur.pdf',dpi=300,bbox_inches='tight')
plt.show()

sns.histplot((df['g']-df['r']),bins='auto',alpha=0.75,stat='density')
plt.xlim(0,1.75)
plt.xlabel('g-r')
plt.savefig('/mnt/sda2/astrometria/practico_3/informe/imagenes/gr.pdf',dpi=300,bbox_inches='tight')
plt.show()  

sns.histplot((df['u']-df['g']),bins='auto',alpha=0.75,stat='density')
plt.xlim(0.25,3.75)
plt.xlabel('u-g')
plt.savefig('/mnt/sda2/astrometria/practico_3/informe/imagenes/ug.pdf',dpi=300,bbox_inches='tight')
plt.show()


#%% ------------------------------------------------------------------
#---------- ROJOS Y AZULES --------------------------------------------------------------
 
cantidad_elip = len(elip)
cantidad_esp = len(esp)
cantidad_mer = len(mer)
data = {
    'Tipo': ['Elípticas', 'Espirales', 'Merger'],
    'Cantidad': [cantidad_elip, cantidad_esp, cantidad_mer]
}
df_tipos = pd.DataFrame(data)
cant= sum(data['Cantidad'])

chi_norm=[]
grados_libertad=2 #cantidad de bines -1  
chi_critico=st.chi2.ppf((1-0.05),grados_libertad)
p_critico=0.05


esperado_uniforme = [cant/3, cant/3, cant/3]
chi2, p_value1 = st.chisquare(data['Cantidad'], esperado_uniforme)

print(" TEST CONTRA DISTRIBUCIÓN UNIFORME:")
print(f"   Observado: {data['Cantidad']}")
print(f"   Esperado:  {[int(x) for x in esperado_uniforme]}")
print(f"   Chi² = {chi2:.4f}, Chi critico = {chi_critico:.4f} ")
print(f"   p-value = {p_value1:.4f}")
print(f"   Grados de libertad = {grados_libertad}")
print('Usando como test estadístico al valor de p')
if p_value1 < p_critico:
    print("   RESULTADO: Rechazamos H₀ - La distribución NO es uniforme")
else:
    print("   RESULTADO: No rechazamos H₀ - La distribución podría ser uniforme")
print('Usando como test estadístico al chi²')
if chi2 >= chi_critico:
    print("   RESULTADO: Rechazamos H₀ - La distribución NO es uniforme")
else:
    print("   RESULTADO: No rechazamos H₀ - La distribución podría ser uniforme")

# Crear el gráfico con seaborn
plt.figure(figsize=(10, 6))
colors = [color_r, color_a, "#8732cd"]  # Rojo, azul, verde

# Crear el barplot
ax = sns.barplot(data=df_tipos, x='Tipo', y='Cantidad',palette=colors, alpha=0.7,hue='Tipo')
ax1 = sns.barplot(data=df_tipos, x='Tipo', y=esperado_uniforme,color='lightsteelblue',alpha=0.1, hatch='/',label='Distribución uniforme')
# Personalizar el gráfico
plt.ylabel('Cantidad de galaxias')
plt.xlabel('Tipo de galaxia')
plt.legend()
plt.title('Distribución de tipos de galaxias (con umbral > 0.8)')
plt.savefig('/mnt/sda2/astrometria/practico_3/informe/imagenes/morfologia.pdf',dpi=300,bbox_inches='tight')



#%% ------------------------------------------------------------------
#---------- ROJOS Y AZULES --------------------------------------------------------------
sns.histplot((rojo['u']-rojo['r']),bins=10,alpha=0.75,stat='density',label='rojas',color=color_r)
sns.histplot((azul['u']-azul['r']),bins=10,alpha=0.75,stat='density',label='azules',color=color_a)
plt.xlim(0.25,4.5)
plt.xlabel('u-r')
plt.show()
sns.histplot((rojo['g']-rojo['r']),bins=100,alpha=0.75,stat='density',label='rojas',color=color_r)
sns.histplot((azul['g']-azul['r']),bins=100,alpha=0.75,stat='density',label='azules',color=color_a)
plt.xlim(0,1.75)
plt.xlabel('g-r')
plt.show()
sns.histplot((rojo['u']-rojo['g']),bins=100,alpha=0.75,stat='density',label='rojas',color=color_r)
sns.histplot((azul['u']-azul['g']),bins=100,alpha=0.75,stat='density',label='azules',color=color_a)
plt.xlim(0.25,3.75)
plt.xlabel('u-g')
plt.show()


#%% ------------------------------------------------------------------
#---------- BULGE Y DISCO ----------------------------------------------------------------

sns.histplot(bulge['elip'],bins='auto',alpha=0.75,stat='density')
sns.histplot(disco['elip'],bins='auto',alpha=0.75,stat='density')
plt.xlim(0,1)
plt.xlabel('elipticas')
plt.show()
sns.histplot(bulge['esp'],bins='auto',alpha=0.75,stat='density')
sns.histplot(disco['esp'],bins='auto',alpha=0.75,stat='density')
plt.xlim(0,1)
plt.xlabel('espirales')
plt.show()

#---------- ESPIRALES Y ELIPTICAS --------------------------------------------------------

sns.histplot(esp['fracDeV_r'],bins='auto',alpha=0.75)
plt.xlim(0,1)
plt.xlabel('fracDeV')
plt.show()


sns.histplot(elip['fracDeV_r'],bins=10,alpha=0.75)
plt.xlim(0,1)
plt.xlabel('fracDeV')
plt.show()

#---------- AJUSTE LINEAL DE UNA MUESTRA --------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=muestra['g'],y=muestra['r'], alpha=0.6, label='Datos originales')
plt.plot(muestra['g'], y, color=color_r, alpha=0.8, label=f'Ajuste: r = {a:.3f}g + {b:.3f}')
plt.ylim(14.25,17.65)
plt.xlabel('g')
plt.ylabel('r')
plt.legend()
plt.title('Ajuste lineal g vs r')
plt.savefig('/mnt/sda2/astrometria/practico_3/informe/imagenes/ajustelineal.pdf',dpi=300,bbox_inches='tight')
plt.show()
 



