import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
import uncertainties.umath as umath
from scipy import integrate

# create build directory
import os
if os.path.exists("build") == False:
   os.mkdir("build")

# boltzmann constant
k = const.k
k_ev = const.k/const.e

# exponential curve
def exp(T, a, b):
    return a*np.exp(-b/T)

### data
# extract data
t, T, I, dI = np.genfromtxt('../data/step15.csv', unpack=True)
# adjust data
T += 273.15
I *= 10**(-11) 
dI *= 10**(-11)
I = unp.uarray(I, dI)

### Heizrate
params, covariance_matrix = np.polyfit(t/60, T, deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))
b = ufloat(params[0], errors[0])
T_0 = ufloat(params[1], errors[1])
print('Heizraten-Fit:')
print(f'b = {b:uS}, T0_1= {T_0:uS}')


### background
# fit 
bgr_I = np.concatenate((I[0:12],I[45:49]), axis=None)
bgr_T = np.concatenate((T[0:12],T[45:49]), axis=None)
p_bgr, cov_bgr = curve_fit(exp, bgr_T, unp.nominal_values(bgr_I))
err_bgr = np.sqrt(np.diag(cov_bgr))
# print results
print('Untergrund-Fit Messung:')
print(f'a = {ufloat(p_bgr[0],err_bgr[0]):.3uS}')
print(f'b = {ufloat(p_bgr[1],err_bgr[1]):.3uS}')
print('\n')
# correct data
I_old = I
I = I - exp(T, p_bgr[0], p_bgr[1])
I_raw = I[16:46]
T_raw = T[16:46]
T_inv = 1/T_raw


integral = np.zeros(len(T_raw))
for i in range(len(T_raw)):
    integral[i] = integrate.simps(unp.nominal_values(I_raw[i:]), T_raw[i:]) / (b.nominal_value * I_raw[i].nominal_value)
#integral = integrate.simps(unp.nominal_values(I_raw), unp.nominal_values(T_raw)) / (b.nominal_value * unp.nominal_values(I_raw))

def y(T, a , b):
    return a/T + b
params, cov_matrix = np.polyfit(T_inv[:-1], np.log(integral[:-1]), deg=1, cov=True)
p = unp.uarray(params, np.sqrt(np.diag(cov_matrix)))

print(f'a = {p[0]*k_ev:.2uS} \n b = {p[1]:.2uS}')

x = np.linspace(T_inv[0], T_inv[-1], 100)
fig, ax = plt.subplots()
ax.plot(x, params[0]*x + params[1], color='mediumblue', label='linearer Fit')
ax.plot(T_inv[:-1], np.log(integral[:-1]), '+', color='royalblue', label='Messwerte')
ax.set_xlabel(r'$1/T/1/\si{\kelvin}$')
ax.set_ylabel(r"$\ln \left( \frac{\int_T^{\infty} I(T')\mathrm{d}T'}{b \cdot I(T)} \right)$")
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
fig.savefig('build/plot2_2.pdf')
fig.clf()
