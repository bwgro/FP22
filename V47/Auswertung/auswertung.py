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



#Funktionen
def fit(x,y,deg):
    fit = np.polyfit(x, y, deg)
    return np.polyval(fit, x)

def Temperatur(R):
    T = 0.00134*R**2 + 2.296*R - 243.02
    return T + 273.15



#Einheiten korriegieren
alpha = alpha*10**(-6)


#Aufgabe b)

C_v = -9*alpha**2*Kompressionsmodul*Molaresvolumen

print(C_v)