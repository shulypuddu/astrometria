#%%
import numpy as np 
import matplotlib.pyplot as plt
import fun2 as f
import scipy.stats as st
from astropy.stats import bootstrap


#%%
#---------------------------- BOOTSTRAP ----------------------------#
n=300
mean=2.0
sigma=1.5
#muestras= st.norm(loc=mean,scale=sigma,size=n,random_state=None)
muestras=np.random.normal(loc=mean,scale=sigma,size=n)
#loc= media de la dist. gaussiana, scale=desvio estandar
#por default son loc=0 scale=1
plt.figure(figsize=(10,6))
plt.hist(muestras,bins='auto',density=True,alpha=0.5,color='g')
plt.plot()
print("Media muestral: ",np.mean(muestras))
print("Varianza muestral: ",np.var(muestras,ddof=1))


def bootstrap(x, func, m=1000):
    """
    x=datos
    func=estadistico que quiero calcular
    m=cantidad de veces que se va a realizar el bootstrap

    """
    _y = np.zeros(m)
    for i in range(m):
        _x=np.random.choice(x,size=len(x),replace=True)
        _y[i]= func(_x)

    return _y 

m=10000
y= bootstrap(muestras,np.mean,m)
delta_y=bootstrap(muestras,np.std,m)
z= (y - muestras.mean())/(muestras.std()/np.sqrt(n))

print(y.mean())
x = np.linspace(min(y), max(y), 1000)
media = np.mean(y)
desvio = np.std(y)

plt.figure(figsize=(10,6))
plt.hist(y,bins='auto',density=True,alpha=0.5,color='g')
plt.plot(x,st.norm.pdf(x, media, desvio), 'r-', lw=2, label='Gaussiana teórica')

y1=st.norm.ppf(0.025,loc=media,scale=desvio) # calcula el percentil 2.5
y2=st.norm.ppf(0.975,loc=media,scale=desvio)
plt.axvline(x=y1,color='b',linestyle='--')
plt.axvline(x=y2,color='b',linestyle='--')
plt.show()



plt.figure(figsize=(10,6))
plt.hist(delta_y,bins='auto',density=True,alpha=0.5,color='b')
plt.plot()


# %%
#---------------------------- CHI CUADRADO ----------------------------#
n =10
p=0.4
x = np.arange(n+1)
binom = st.binom(n, p)
y= binom.pmf(x)

m=100
muestra = np.random.binomial(n, p, size=m)

def chi2(O,E):
    """
    O: de la muestra
    E: de la distribución teórica
    """
    test= np.sum((O-E)**2/E)
    return test


plt.hist(y,bins='auto')

