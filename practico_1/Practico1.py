#%%
import matplotlib.pyplot as plt
import funciones as f
import numpy as np
import time 
#PRIMERA CELDA CON TODAS LAS LIBRERIAS A USAR

#Valores dados por Numerical Recipes para la funcion glc
#me asegura un gran rango de números aleatorios antes de q estos fallen
m=2**32	
a=1664525
c=1013904223

#%%

#------------------- (GLC) Ejercicio 18 Parte (a) -------------------

# Genero numeros aleatorios
N=10000 #cantidad de numeros a generar

semilla =10
numeros = f.glc(N,semilla)

x = numeros[:-1] #toma toda la lista menos el ultimo elemento
y = numeros[1:] #toma toda la lista menos el primer elemento
plt.plot(x,y,'D',color='darkorchid',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Correlación entre pares sucesivos')
plt.legend()
plt.show()

print('Periodo y momentos de los números generados con GLC:')
print(f.periodo(numeros))
print(f'Momento k=1: {f.momentok(1,numeros)}')
print(f'Momento k=3: {f.momentok(3,numeros)}')
print(f'Momento k=7: {f.momentok(7,numeros)}')
print(1/(3+1),1/(7+1))



#---- Repito el procedimiento con numeros mas bonitos
nros = f.glc(N,semilla,a,c,m)

x = nros[:-1] #toma toda la lista menos el ultimo elemento
y = nros[1:] #toma toda la lista menos el primer elemento
plt.plot(x,y,'.',color='lightgreen',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Correlación entre pares sucesivos')
plt.legend()
plt.show()

print('Periodo y momentos de los números generados con GLC:')
print(f.periodo(nros))
print(f'Momento k=1: {f.momentok(1,nros)}')
print(f'Momento k=3: {f.momentok(3,nros)}')
print(f'Momento k=7: {f.momentok(7,nros)}')
print(1/(3+1),1/(7+1))

#%%
#------------------- Ejercicio 18 Parte (b) -------------------

N_caminatas = 10
N_pasos = 1000
#zeros crea un array de numpy con las dimensiones que le paso por parámetro, en este caso 10 filas y 1000 columnas
x= np.zeros((N_caminatas , N_pasos))
y= np.zeros((N_caminatas , N_pasos))
# caminatas == filas pasos== columnas
for i in range(N_caminatas):
    for j in range(N_pasos):
        salto_x= np.random.rand()*2*np.sqrt(2)-np.sqrt(2)
        x[i,j]= salto_x + x[i,j-1]
        salto_y= np.random.rand()*2*np.sqrt(2)-np.sqrt(2)
        y[i,j]= salto_y + y[i,j-1] 

plt.figure(figsize=(10, 8))
for i in range(N_caminatas):
    plt.plot(x[i], y[i],'-',label=f'camino numero {i}')
    #plt.plot(x[i,:], y[i,:],'-',label=f'camino numero {i}') es lo mismo
plt.title(f'Caminatas Aleatorias')
plt.xlabel('$\Delta X$')
plt.ylabel('$\Delta Y$')
#plt.legend()
plt.show()

distances = np.sqrt(x**2 + y**2)
media=np.mean(distances,axis=0) #axis=0 calcula la media a lo largo de las filas, es decir, para cada paso toma las 10 distancias y calcula la media
#distances.mean(axis=0) es lo mismo que np.mean(distances,axis=0)
plt.figure(figsize=(10, 8))
for i in range(N_caminatas):
    plt.plot(distances[i],'-')
plt.plot(media,'k-',linewidth=3,label='media')
plt.xlabel('$Distancia$')
plt.ylabel('$Pasos$')
plt.legend()
plt.show()

#grafico la media en funcion de la raiz cuadrada del paso
plt.figure(figsize=(10, 8))
for i in range(N_caminatas):
    plt.plot(np.sqrt(range(N_pasos)),distances[i],'-')
plt.plot(np.sqrt(range(N_pasos)),media,'k-',linewidth=3,label='media')
plt.xlabel('$\sqrt{N pasos}$')
plt.ylabel('$Distancia$')
plt.title('Distancia media en función de la raíz cuadrada del paso')
plt.legend()
plt.show()



#%%
#---------------------- Ejercicio 19 ----------------------  

x= f.fib(10000)
_x =x[1:]
_y = x[:-1]
#Genero un segundo bloque pues f.fib demora mucho en correr con tantos nros
#%%
plt.plot(_x,_y,'.',color='hotpink',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Fibonacci')
plt.legend()
plt.show()

print('La media de los números generados con Fibonacci es:',np.mean(x))
print('La varianza de los números generados con Fibonacci es:',np.var(x))

plt.figure(figsize=(8, 5))
plt.hist(x, bins=np.arange(0, 1.1, 0.1), color='orange', rwidth=0.8)
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de generador de Fibonacci')
plt.show()

#%%
#Ahora repito usando np.random
y= np.random.random(10000)
xx =x[1:]
yy = x[:-1]

plt.plot(xx,yy,'.',color='orange',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Correlación entre pares sucesivos')
plt.legend()
plt.show()

print('La media de los números generados con Numpy.random es:',np.mean(x))
print('La varianza de los números generados con Numpy.random es:',np.var(x))

plt.figure(figsize=(8, 5))
plt.hist(y, bins=np.arange(0, 1.1, 0.1), color='hotpink', rwidth=0.8)
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de Numpy.random')
plt.show()


#%%
#---------------------- Ejercicio 20 ----------------------  

#Dejo la funcion pearson_correlation junto con el resto de mis funciones

fib_x = f.fib_int(100,100)
fib_y = f.fib_int(100,10)
glc_x = f.glc_int(100,10)
glc_y = f.glc_int(100,100)

# Lista de retardos
retardos = [1, 2, 3, 5, 7, 10]

for i in retardos:
    x_glc = glc_x[:-i]   # quita los últimos d elementos
    y_glc = glc_y[i:]    # quita los primeros d elementos
    p_glc = f.pearson_correlation(x_glc, y_glc)

    x_fib = fib_x[:-i]
    y_fib = fib_y[i:]
    p_fib = f.pearson_correlation(x_fib, y_fib )
    print(f'La correlación de Pearson con retardo {i} para GLC es: {p_glc}')
    print(f'La correlación de Pearson con retardo {i} para el generador de Fibonacci es: {p_fib}')



# %%
#---------------------- Ejercicio 21 ----------------------  
def Monty_Hall_sin_cambiar(n):
    """
    La función simula el juego de Monty Hall sin cambiar de puerta.
    --Paramámetros--
    n: número de veces que se repite el experimento
    
    --Retorna--
    Imprime "Ganó" si el jugador gana el auto, "Perdió" si no.
    
    """
    for i in range(n):
        vector = ['Cabra','Cabra','Auto']
        np.random.permutation(vector)
        eleccion = f.glc_int(1,(i+i**3),a,c,m)[0]%3
        if vector[eleccion] == 'Auto':
            print('Ganó')
        else:
            print('Perdió')
    

def Monty_Hall_cambiando(n):
    """
    La función simula el juego de Monty Hall cambiando de puerta.
    --Paramámetros--
    n: número de veces que se repite el experimento
    --Retorna--
    Imprime "Ganó" si el jugador gana el auto, "Perdió" si no.
    
    """
    return n+1


#%%
#---------------------- Ejercicio 22 ----------------------  

def galaxias(N):
    """
    Genera N galaxias y las clasifica tipos “elíptica”, “espiral”, “enana” e “irregular”
    según la probabilidad de estas.
    0.4, 0.3, 0.2, y 0.1
    """
    elip = 0
    espiral = 0
    enan = 0
    irr = 0
    semilla=int(time.time())
    nros = f.glc(N,semilla,a,c,m)
    for i in range(N): 
        if nros[i]<0.4 :
           elip +=1
        elif nros[i]<0.7 :
            espiral +=1
        elif nros[i]<0.9 :
            enan+=1
        else :
            irr +=1
     
    return elip, espiral, enan, irr

galaxia = galaxias(1000)
print(galaxia)

# calculo los porcentajes de los tipos de galaxias aleatoriamente generadas

elip = galaxia[0]/1000*100
espiral = galaxia[1]/1000*100
enan = galaxia[2]/1000*100
irr = galaxia[3]/1000*100

plt.figure(figsize=(8,5))
plt.bar(['Elíptica','Espiral','Enana','Irregular'],[elip,espiral,enan,irr],color='lightblue',edgecolor='skyblue')
plt.xlabel('Tipo de galaxia')
plt.ylabel('Porcentaje')
plt.title('Porcentaje de tipos de galaxias generadas')
plt.show()

#%%
#---------------------- Ejercicio 23 ----------------------  

var_al=np.arange(2,13)
dist_probabilidad=[1/36,1/18,1/12,1/9,5/36,1/6,5/36,1/9,1/12,1/18,1/36]


plt.figure(figsize=(8,5))
plt.bar(var_al,dist_probabilidad,width=1,color='lawngreen',edgecolor='limegreen')
plt.xlabel('Variable aleatoria')
plt.ylabel('Probabilidad teórica')
plt.show()

n = 10000
# Generamos valores para ambos dados
dado_1 = f.dados(n,252)
dado_2 = f.dados(n,255)
# Sumamos los valores
suma = np.zeros(n)
for i in range(n):
  suma[i] = dado_1[i] + dado_2[i]

# Distribución empírica
suma_empirica, frec_empirica = np.unique(suma,return_counts=True)
prob_empirica = frec_empirica / n

# Comparar la distribución empírica con la teórica
plt.figure(figsize=(10, 6))
plt.bar(var_al - 0.2, dist_probabilidad, width=0.4, label='Teórica', color='lawngreen')
plt.bar(suma_empirica + 0.2, prob_empirica, width=0.4, label='Empírica', color='hotpink')
plt.xlabel('Suma de los dados')
plt.ylabel('Probabilidad')
plt.title('Comparación entre la distribución teórica y empírica')
plt.legend()
plt.grid(True)
plt.show()


# Generamos valores para ambos dados
_dado = f.dado_doble(n)


# Distribución empírica
suma2_empirica, frec2_empirica = np.unique(_dado,return_counts=True)
prob2_empirica = frec2_empirica / n

# Comparar la distribución empírica con la teórica
plt.figure(figsize=(10, 6))
plt.bar(var_al - 0.2, dist_probabilidad, width=0.4, label='Teórica', color='lawngreen')
plt.bar(suma2_empirica + 0.2, prob2_empirica, width=0.4, label='Empírica', color='hotpink')
plt.xlabel('Usando dado doble (2-12)')
plt.ylabel('Probabilidad')
plt.title('Comparación entre la distribución teórica y empírica')
plt.legend()
plt.grid(True)
plt.show()



#%%

# np.random.permutation() permuta los elementos de un array de forma aleatoria
# la lista puede contener elementos tipos string

