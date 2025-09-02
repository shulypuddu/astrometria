#%%
import matplotlib.pyplot as plt
from funciones import fib 
#no hace falta importar las funciones con las q trabaja la funcion fib 


def ej19f1():
    x= fib_int(144152,1000)
    _x =x[1:]
    _y = x[:-1]

    plt.plot(_x,_y,'x',label='pares')
    plt.xlabel('$n_{i}$')
    plt.ylabel('$n_{i+1}$')
    plt.title('$Titulo$')
    plt.legend()
    plt.show()

ej19f1()


# %%
