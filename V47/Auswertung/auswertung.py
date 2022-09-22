from cProfile import label
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from codecs import open
from uncertainties import unumpy as unp
from scipy.integrate import quad

#Konstanten
Masse = 0.342
Molmasse = 63.546
Dichte = 8.92*10**6
Molaresvolumen = 7.11*10**-6
Kompressionsmodul = 140*10**9



#Einlesen der Daten
R_p, R_g, I, U, t = np.genfromtxt('Messdaten/Messdaten.txt', unpack=True)
T, alpha = np.genfromtxt('Messdaten/alpha.txt', unpack=True)



#Funktionen#####################################
#Wiederstände in Temperatur
def Temperatur(R):
    T = 0.00134*R**2 + 2.296*R - 243.02
    T = T + 273.15
    return T



#Einheiten################################################ 
alpha = alpha*10**(-6) #Größenordnung korrigieren
M = 63.546 #Molare Masse
m = 342.00 #Masse der Probe in Gramm
Mol_V = 7.11*10**-6 #Molares Volumen
Roh = 8.92*10**6 #Dichte
kappa = 140*10**9



#Berechnung C_p##########################
C_p = [0] * 22
i = 0
while i <= 21:
    E = U[i] * I[i]/1000 * ((t[i+1] - t[i])*60)
    C_p[i] = M/m * E/(Temperatur(R_p[i+1]) - Temperatur(R_p[i]))
    i = i + 1


#Berechnung C_v#########################################
#alpha Plotten
a = np.polyfit(T, alpha,4)
a = np.poly1d(a)

C_v = [0] * 23
i = 0
while i <= 21:
    C_v[i] = C_p[i] - 9*(a(t[i])**2)*kappa*Mol_V*t[i]
    i = i+1


ausgleichsgeradefit = np.polyfit(Temperatur(R_p),C_v,1)
ausgleichsgerade = np.poly1d(ausgleichsgeradefit)





#Plots###################################################################

#C_V
plt.scatter(Temperatur(R_p),C_v, marker = 'x', label = 'C_v')
#plt.plot(Temperatur(R_p),ausgleichsgerade(Temperatur(R_p)), color = 'r', label = 'Ausgleichsgerade durch die C_v Werte')
plt.xlabel('Temperatur[K]')
plt.ylabel('C_v[J/mol*K]')
plt.grid()
plt.legend()
plt.savefig('Plots/C_v.pdf')
plt.clf()

#alpha
plt.scatter(T,alpha, label = 'Messwerte')
plt.plot(T,a(T), label = 'fit')
plt.xlabel('Temperatur[K]')
plt.ylabel('alpha')
plt.grid()
plt.legend()
plt.savefig('Plots/alpha.pdf')

