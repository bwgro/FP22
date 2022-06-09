import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op

#Gauß-Distribution
def normal_distribution(x, integral, mean, sigma):
    return integral / np.sqrt(2 * np.pi) / sigma *np.exp(- x**2 / sigma**2)

#Messdaten einlesen
x,y = np.genfromtxt('Messwerte/Detektorscan.txt', unpack=True)

#Auswertung der Funktion
popt, pconv = op.curve_fit(normal_distribution, x,y)
fitted_x = np.linspace(np.min(x), np.max(x), 1000)
fitted_y = normal_distribution(fitted_x, *popt)

Amplitude = popt[0]
Mittelwert = popt[1]
Standartabweichung = 2.4 * popt[2]

#Plot
pl.plot(x, y,'o', linestyle = 'none', label='Messdaten')
pl.plot(fitted_x, fitted_y, linestyle='-', label = 'Ausgleichskurve')
pl.xlabel('\u03B1/°')
pl.ylabel("Intensität")
pl.title("Detektorscan")
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/Detektorscan.pdf')

#Messdaten
print('Amplitude:', Amplitude)
print('Mittelwert:', Mittelwert)
print('Halbwertsbreite', Standartabweichung)
