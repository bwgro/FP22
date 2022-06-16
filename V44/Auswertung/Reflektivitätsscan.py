from turtle import color
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op
import math as ma





#Import Daten
x,y = np.genfromtxt('Messwerte/omega_tet.txt', unpack=True)
a,b = np.genfromtxt('Messwerte/diffusor.txt' , unpack=True)



#Reflektivität
I_0 = 1114508.0792544584
R_ref = y / (5 * I_0)
R_diff = b / (5 * I_0)
R= R_ref - R_diff



#Geometriefaktor
a_g = 0.4
R_g = np.zeros(np.size(R))

for i in np.arange(np.size(x)):
    if(x[i] <= a_g and x[i] > 0 ):
        R_g[i] = R[i] * np.sin(np.deg2rad(a_g)) / np.sin(np.deg2rad(x[i]))
    else:
        R_g[i] = R[i]



#Frenessel
f = (0.223 / (2 * x) )**4

#Schichtdicke
min_x = [x[69],x[78],x[87],x[97],x[106],x[117]]
min_y = [R[69],R[78],R[87],R[97],R[106],R[117]]
print('Schichtdicke: ',x[87] - x[78])


#Plot
pl.plot(x, R, '-', label='Messdaten')
pl.plot(x, R_g, '-', label='Reflektivität mit Geometriefaktor')
pl.plot(x, f, label='Ideale Silitzium Oberfläche')
pl.scatter(min_x, min_y, marker='o',color = 'red', s=10, label ='Minima')
pl.yscale('log')
pl.xlabel("'\u03B1/°'")
pl.ylabel("R")
pl.title("Reflektivitätsscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/Reflektivitaetsscan.pdf')