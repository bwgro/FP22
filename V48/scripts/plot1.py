import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
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
t, T, I, dI = np.genfromtxt('../data/step20.csv', unpack=True)
# adjust data
T += 273.15
I *= 10**(-11) 
dI *= 10**(-11)

### Heizrate
params, covariance_matrix = np.polyfit(t/60, T, deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))
b = ufloat(params[0], errors[0])
T_0 = ufloat(params[1], errors[1])
print('Heizraten-Fit:')
print(f'b = {b:uS}, T0_1= {T_0:uS}')
print('\n')


### background
# fit 
bgr_I = np.concatenate((I[0:6],I[25]), axis=None)
bgr_T = np.concatenate((T[0:6],T[25]), axis=None)
p_bgr, cov_bgr = curve_fit(exp, bgr_T, bgr_I)
err_bgr = np.sqrt(np.diag(cov_bgr))
# print results
print('Untergrund-Fit Messung:')
print(f'a = {ufloat(p_bgr[0],err_bgr[0]):.3uS}')
print(f'b = {ufloat(p_bgr[1],err_bgr[1]):.3uS}')
print('\n')
# correct data
I_old = I
I = I - exp(T, p_bgr[0], p_bgr[1])
I_raw = I[7:18]
T_raw = T[7:18]
# max temperature
T_max = np.max(T_raw)
# plot
fig, ax = plt.subplots()
x = np.linspace(220, 310 ,100)
ax.plot(x, exp(x, p_bgr[0], p_bgr[1]), color='mediumblue', label = 'Untergrund')
ax.plot(T, I_old, '+', color='royalblue', label=r'Messwerte')
ax.plot(T[9:26], I[9:26], '+r', color='crimson', label=r'korrigierte Messwerte')
ax.set_xlabel(r'$T/\si{\kelvin}$')
ax.set_ylabel(r'$I/\si{\ampere}$')
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
fig.savefig('build/plot1_bgr.pdf')
fig.clf()

### activation energy 1 by polarisation
# relaxation
print(f'Intervall für Aktivierungsenergie:')
print(T_raw[0],T_raw[-1])
print('\n')
# log(I) & 1/T
I_log = np.log(I[10:21])
T_inv = 1/(T[10:21])
# fit linear
p_log, cov_log = np.polyfit(T_inv, I_log, deg=1, cov=True)
err_log = np.sqrt(np.diag(cov_log))
# calculate W
W_1 = -ufloat(p_log[0],err_log[0])*k_ev
# print results
print('Polarisationsansatz linearer Fit:')
print(f'm = {ufloat(p_log[0],err_log[0]):.16uS}')
print(f'b = {ufloat(p_log[1],err_log[1]):.3uS}')
print(f'W = -m*k_B = {W_1:.2uS} [eV]')
print('\n')
# plot
x = np.linspace(T_inv[0], T_inv[-1], 100)
fig, ax = plt.subplots()
ax.plot(x, p_log[0]*x + p_log[1], color='mediumblue', label='linearer Fit')
ax.plot(T_inv, I_log, '+', color='royalblue', label='Messwerte')
ax.set_xlabel(r'$1/T/1/\si{\kelvin}$')
ax.set_ylabel(r'$\ln(I)/\ln(\si{\ampere})$')
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
fig.savefig('build/plot1_1.pdf')
fig.clf()

### activation energy 2 by integration

W_2 = ufloat(0.996, 0.014)
# print results
print('Stromdichtenansatz exp. Fit:')

print(f'W = m*k_B = {W_2:.3uS} [eV]')
print('\n')

### relaxation time
# characteristic
tau_1 = k_ev*T_max**2/b/W_1*unp.exp(-W_1/k_ev/T_max)
tau_2 = k_ev*T_max**2/b/W_2*unp.exp(-W_2/k_ev/T_max)
print('charak. Relaxationszeit:')
print(f'tau_0,1 = {tau_1:.5S} [min] = {tau_1*60:.3S} [s]')
print(f'tau_0,2 = {tau_2:.5S} [min] = {tau_2*60:.3S} [s]')

#temperature-dependent
def tau(T, t0, W):
    return unp.nominal_values(t0) * np.exp(W.nominal_value/k_ev/T)

fig, ax = plt.subplots()
ax.plot(T, tau(T, tau_1, W_1), '+', color='crimson', label='Polarisationsansatz')
ax.plot(T, tau(T, tau_2, W_2), '+', color='royalblue', label='Stromdichtenansatz')
ax.set_xlabel(r'$T/\si{\kelvin}$')
ax.set_ylabel(r'$\ln(\tau)/\ln(\si{\second})$')
ax.set_yscale('log')
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
fig.savefig('build/plot1_t.pdf')
fig.clf()

### literature
W_lit = ufloat(0.66,0.01)
dW = np.abs((W_1-W_lit)/W_lit)
print(dW)

