import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op

x,y = np.genfromtxt('Messwerte/omega_tet.txt', unpack=True)
a,b = np.genfromtxt('Messwerte/diffusor.txt' , unpack=True)

y = y - b

pl.plot(x, y, label='Daten')
pl.yscale('log')
pl.xlabel("Delta")
pl.ylabel("Intensitaet")
pl.title("sReflektivitaetsscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/Reflektivitaetsscan.pdf')