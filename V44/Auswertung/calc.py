import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op

def normal_distribution(x, integral, mean, sigma):
    return integral / np.sqrt(2 * np.pi) / sigma *np.exp(- x**2 / sigma**2)

x,y = np.genfromtxt('Messwerte/Detektorscan.txt', unpack=True)
popt, pconv = op.curve_fit(normal_distribution, x,y)
pl.plot(x,y)
fitted_x = np.linspace(np.min(x), np.max(x), 1000)
fitted_y = normal_distribution(fitted_x, *popt)

pl.plot(x, y, marker='o', label='Daten')
pl.plot(fitted_x, fitted_y, linestyle='-', label = 'Anpassung')
pl.grid(True)
pl.legend(loc='best')
pl.savefig('Graphen/test.pdf')
