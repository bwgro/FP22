from cProfile import label
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op
import math as ma

#Messdaten einlesen
x,y = np.genfromtxt('Messwerte/z_Scan.txt', unpack=True)
a,b = np.genfromtxt('Messwerte/Rocking_0.txt', unpack=True)

#Berechnung Strahlenbreite
Strahlenbreite = x[37] - x[27]
print('Strahlenbreite: d_0=', Strahlenbreite)

#Geometriewinkel
print('Strahlenwinkel: a_g=', x[45])


#Plot Z-Scan
pl.plot(x, y, linestyle='-', label = 'Messdaten')
pl.axvline(x[37], color = 'g' , linestyle='--', label = 'Strahlengrenzen')
pl.axvline(x[27], color = 'g' , linestyle='--')
pl.xlabel("z/mm")
pl.ylabel("Intensitaet")
pl.title("Detektorscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/z_Scan.pdf')
pl.clf()

#Plot Rockingscan
pl.plot(a, b, linestyle='-', label = 'Messdaten')
pl.axvline(x[45], color = 'g' , linestyle='--', label = 'Geometriewinkel')
pl.axvline(x[3], color = 'g' , linestyle='--')
pl.xlabel('\u03B1/°')
pl.ylabel("Intensitaet")
pl.title("Detektorscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/Rocking_Scan.pdf')
pl.clf()



