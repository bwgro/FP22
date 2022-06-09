import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op
import math as ma

#Freneschel Formel
def fresnel(alpha):
    n_1 = 1
    n_2 = 3.35
    beta = ma.asin(n_1/n_2 * ma.sin(alpha))
    return (ma.tan(alpha - beta))/(ma.tan(alpha + beta))

def Korrektur(a_g):
    return (((0.02*ma.sin(a_g))/0.2))



#Import Daten
x,y = np.genfromtxt('Messwerte/omega_tet.txt', unpack=True)
a,b = np.genfromtxt('Messwerte/diffusor.txt' , unpack=True)


#Auswertung
r = y - b
k = r
f = y
i = 0
j = 1

while x[i] <=0.4:
    k[i] = Korrektur(x[i])*y[i]
    i = i+1

while j <=500:
    alpha = x[j]
    f[j] = fresnel(x[j])
    j = j+1
    




#Plot
pl.plot(x, r, label='Messdaten')
pl.plot(x, k, label='Reflektivität mit Geomitriefaktor')
pl.plot(x, f, label='Ideale Silitzium Oberfläche')
pl.plot
pl.yscale('log')
pl.xlabel("'\u03B1/°'")
pl.ylabel("R")
pl.title("Reflektivitätsscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/Reflektivitaetsscan.pdf')