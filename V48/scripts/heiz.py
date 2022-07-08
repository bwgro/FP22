import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy import integrate

import os

if os.path.exists("build") == False:
   os.mkdir("build")

#konstante definieren
k = const.k
k_ev = const.k/const.e

#Werte auslesen
t_1, T_1, I_1, dI_1 = np.genfromtxt('../data/step20.csv', unpack=True)

T_1 += 273.15 # in kelvin
I_1 *= 10**(-11) #pico ampere
dI_1 *= 10**(-11)

t_2, T_2, I_2, dI_2 = np.genfromtxt('../data/step15.csv', unpack=True)

T_2 += 273.15 # in kelvin
I_2 *= 10**(-11) #pico ampere
dI_2 *= 10**(-11)

# Heizrate
t_1 *= 1/60 # sec to min
t_2 *= 1/60 # sec to min

params1, covariance_matrix1 = np.polyfit(t_1, T_1, deg=1, cov=True)
params2, covariance_matrix2 = np.polyfit(t_2, T_2, deg=1, cov=True)
eParams1 = unp.uarray(params1, np.sqrt(np.diag(covariance_matrix1)))
eParams2 = unp.uarray(params2, np.sqrt(np.diag(covariance_matrix2)))

print('Heizraten:')
print(f'b_1 = {eParams1[0]:uS}, T0_1= {eParams1[1]:uS}')
print(f'b_2 = {eParams2[0]:uS}, T0_2= {eParams2[1]:uS}')

z = np.linspace(0,76,100)
plt.plot(z[:-25], params1[0]*z[:-25] + params1[1], color='mediumblue', label=r'Ausgleichsgerade 1')
plt.plot(t_1, T_1, '+', color='royalblue' ,label=r'Messung 1')
plt.plot(z, params2[0]*z + params2[1], color='darkslategrey', label=r'Ausgleichsgerade 2')
plt.plot(t_2, T_2, '+', color='crimson', label=r'Messung 2')

plt.xlabel(r'$t/\si{\minute}$')
plt.ylabel(r'$T/\si{\kelvin}$')
plt.tight_layout()
plt.grid()
plt.legend(loc='best')
plt.savefig('build/Heizrate.pdf')
plt.clf()