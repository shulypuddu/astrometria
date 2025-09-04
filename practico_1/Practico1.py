#%%
import matplotlib.pyplot as plt
import funciones as f
import numpy as np
import time 
#PRIMERA CELDA CON TODAS LAS LIBRERIAS A USAR

#%%

#------------------- Ejercicio 18 Parte (a) -------------------

# Genero numeros aleatorios
semilla =10
numeros = f.glc(1000,semilla)

x = numeros[:-1] #toma toda la lista menos el ultimo elemento
y = numeros[1:] #toma toda la lista menos el primer elemento
plt.plot(x,y,'D',color='darkorchid',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Correlación entre pares sucesivos')
plt.legend()
plt.show()


print(f.periodo(numeros))
print(f.momentok(1,numeros))
print(f.momentok(3,numeros))
print(f.momentok(7,numeros))
print(1/(3+1),1/(7+1))
#---- Repito el procedimiento con numeros mas bonitos


#Valores dados por Numerical Recipes para la funcion glc, me asegura un gran rango de números aleatorios antes de q estos fallen

m=2**32	
a=1664525
c=1013904223

nros = f.glc(10,semilla,a,c,m)

x = nros[:-1] #toma toda la lista menos el ultimo elemento
y = nros[1:] #toma toda la lista menos el primer elemento
plt.plot(x,y,'.',color='lightgreen',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('Correlación entre pares sucesivos')
plt.legend()
plt.show()


print(f.periodo(nros))
print(f.momentok(1,nros))
print(f.momentok(3,nros))
print(f.momentok(7,nros))
print(1/(3+1),1/(7+1))

#%%
#------------------- Ejercicio 18 Parte (b) -------------------

def caminos(N,k,x0):
    """
    Crea k caminos con N cantidad de pasos

    
    --Parámetros--
    Todos los parámetros son enteros
    N : cantidad de pasos (cant de valores en x e y) que va a tener el camino 
    k : cantidad de caminos
    x0 : semilla inicial (solo para generar las semillas iniciales)


    --Retorna--
    N caminos en un rango de -sqrt(2) a sqrt(2)
    """
    pasos = []
    semillas = f.glc(k,x0,a,c,m)

    for j in range(k):
        dx = []
        dy = []
        x = f.glc(N, a, c, m, semillas[j])
        y =f.glc(N,a, c, m, semillas[j-1])
        for i in range(N):
            dx.append(2 * (x[i] - 0.5) * np.sqrt(2))
            dy.append(2 * (y[i] - 0.5) * np.sqrt(2))
        X = np.cumsum(dx)
        Y = np.cumsum(dy)
        pasos.append((X, Y))
    return pasos

def valor_exp_camino():
    return 0

caminos = caminos(1000,10,252)
plt.figure(figsize=(10, 8))
for i, (X, Y) in enumerate(caminos):
    plt.plot(X, Y,'--',label=f'camino numero {i}')
    plt.scatter(X[0], Y[0])
    plt.scatter(X[-1], Y[-1])
plt.title(f'Caminatas Aleatorias')
plt.xlabel('$\Delta X$')
plt.ylabel('$\Delta Y$')
plt.grid(True)
plt.show()


#%%
#---------------------- Ejercicio 19 ----------------------  
def ej19f1():
    x= f.fib(130,16540)
    _x =x[1:]
    _y = x[:-1]

    plt.plot(_x,_y,'x',label='pares')
    plt.xlabel('$n_{i}$')
    plt.ylabel('$n_{i+1}$')
    plt.title('$Titulo$')
    plt.legend()
    plt.show()
    print(x)

ej19f1()

y= np.random.random(10)
print(y)






# %%
#---------------------- Ejercicio 20 ----------------------  










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


dado_1 = dados(10,int(time.time()))
dado_2 = dados(10,int(time.time())-3)
print(dado_1)
print(dado_2)





