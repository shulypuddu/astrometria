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
print(f.momentok(1,numeros))
print(f.momentok(3,numeros))
print(f.momentok(7,numeros))
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
print(f.momentok(1,nros))
print(f.momentok(3,nros))
print(f.momentok(7,nros))
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
    plt.plot(distances[i],'-',label=f'camino numero {i}')
plt.plot(media,'k-',linewidth=3,label='media')
plt.xlabel('$Distancia$')
plt.ylabel('$Pasos$')
#plt.legend()
plt.show()




#%%
#---------------------- Ejercicio 19 ----------------------  
def ej19f1():
    x= f.fib(130,2000)
    _x =x[1:]
    _y = x[:-1]

    plt.plot(_x,_y,'x',label='pares')
    plt.xlabel('$n_{i}$')
    plt.ylabel('$n_{i+1}$')
    plt.title('Correlación entre pares sucesivos')
    plt.legend()
    plt.show()
   

ej19f1()

y= np.random.random(10)
print(y)

#%%
#---------------------- Ejercicio 20 ----------------------  

def pearson_correlation (x,y):
    """
    Calcula el coeficiente de correlacion de Pearson entre dos arrays
    Parameters :
    x , y : arrays de igual longitud
    Returns :
    r : coeficiente de correlacion de Pearson
    """
    # Verificar que tienen la misma longitud
    if len ( x ) != len ( y ) :
        raise ValueError ( " Los arrays deben tener la misma longitud " )
    n = len ( x )
    # Calcular medias
    mean_x = np . mean ( x )
    mean_y = np . mean ( y )
    # Calcular numerador y denominador
    numerator = np . sum (( x - mean_x ) * ( y - mean_y ) )
    denominator = np . sqrt ( np . sum (( x - mean_x ) **2) * np . sum (( y - mean_y ) **2) )
    # Evitar division por cero
    if denominator == 0:
        return 0
    return numerator / denominator

fib_x = f.fib_int(100,100)
fib_y = f.fib_int(100,10)
glc_x = f.glc_int(100,10)
glc_y = f.glc_int(100,100)
print(pearson_correlation(fib_x,fib_y))
print(pearson_correlation(glc_x,glc_y)) 



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
   return 0





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

print(f'El porcentaje de galaxias elípticas es {elip} %')
print(f'El porcentaje de galaxias espirales es {espiral} %')
print(f'El porcentaje de galaxias enanas es {enan} %')      
print(f'El porcentaje de galaxias irregulares es {irr} %')


#%%
#---------------------- Ejercicio 23 ----------------------  

def dados(N,n0):
    """
    --Parámetros--
    N : cantidad de tiradas
    n0 : semilla inicial
    
    --Retorna--
    lista de N valores entre 0 y 6
    """
    _nros= f.glc_int(N,n0,a,c,m)
    nros = np.array(_nros)%6
    for i in range(len(nros)):
        if nros[i]== 0:
            nros[i]=6
    return nros


dado_1 = dados(100,int(time.time()))
dado_2 = dados(100,int(time.time())-3)

#Ahora sumo los resultados de ambos dados
suma_dados = dado_1 + dado_2
print(suma_dados)





#%%

# np.random.permutation() permuta los elementos de un array de forma aleatoria
# la lista puede contener elementos tipos string

