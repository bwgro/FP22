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
wl = 1.54e-10           #Wellenlänge
min_x = [x[67],x[78],x[87],x[97],x[106],x[117]]
min_y = [R[67],R[78],R[87],R[97],R[106],R[117]]
print('Minimum werte:', min_x, min_y)
i = 0
d = [0,0,0,0,0]
while i <= 4:
    d[i] = min_y[i] - min_y[i + 1]
    i = i + 1
Schichtdicke_min = np.mean(d)
Abweichung_min = np.std(d)
print('delta Alpha:', Schichtdicke_min)
print('Schichtdicke Min: ',wl/(2*Schichtdicke_min))
print('Standartabweichung Schichtdicke Min:', Abweichung_min)



#Plot
pl.ylim(1*10**-8, 1*10*1)
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