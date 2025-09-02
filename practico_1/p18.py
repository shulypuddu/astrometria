import matplotlib.pyplot as plt

#-------- Parte (a) --------

        # Genero numeros aleatorios
semilla =10
numeros = glc(10,semilla)

x = numeros[:-1] #toma toda la lista menos el ultimo elemento
y = numeros[1:] #toma toda la lista menos el primer elemento
plt.plot(x,y,'x',label='pares')
plt.xlabel('$n_{i}$')
plt.ylabel('$n_{i+1}$')
plt.title('$Titulo$')
plt.legend()
plt.show()

plt.hist(numeros)
plt.title('Frecuencia de los números generados')
plt.xlabel('Número')
plt.ylabel('Frecuencia')
plt.show()

#-------- Parte (b) --------









