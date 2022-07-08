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
t, T, I, dI = np.genfromtxt('../data/step15.csv', unpack=True)
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


### background
# fit 
bgr_I = np.concatenate((I[0:12],I[45:49]), axis=None)
bgr_T = np.concatenate((T[0:12],T[45:49]), axis=None)
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
I_raw = I[18:35]
T_raw = T[18:35]
# max temperature
T_max = np.max(T_raw)
# plot
fig, ax = plt.subplots()
x = np.linspace(220, 310 ,100)
ax.plot(x, exp(x, p_bgr[0], p_bgr[1]), color='mediumblue', label = 'Untergrund')
ax.plot(T, I_old, '+', color='royalblue', label=r'Messwerte')
ax.plot(T[11:46], I[11:46], '+r', color='crimson', label=r'korrigierte Messwerte')
ax.set_xlabel(r'$T/\si{\kelvin}$')
ax.set_ylabel(r'$I/\si{\ampere}$')
ax.legend(loc='best')
ax.grid()
fig.tight_layout()
fig.savefig('build/plot2_bgr.pdf')
fig.clf()

### activation energy 1 by polarisation
# relaxation
print(f'Intervall für Aktivierungsenergie:')
print(T_raw[0],T_raw[-1])
print('\n')
# log(I) & 1/T
I_log = np.log(I_raw)
T_inv = 1/(T_raw)
# fit linear
p_log, cov_log = np.polyfit(T_inv, I_log, deg=1, cov=True)
err_log = np.sqrt(np.diag(cov_log))
# calculate W
W_1 = -ufloat(p_log[0],err_log[0])*k_ev
# print results
print('Polarisationsansatz linearer Fit:')
print(f'm = {ufloat(p_log[0],err_log[0]):.3uS}')
print(f'b = {ufloat(p_log[1],err_log[1]):.3uS}')
print(f'W = -m*k_B = {W_1:.16uS} [eV]')
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
fig.savefig('build/plot2_1.pdf')
fig.clf()

### activation energy 2 by integration
# integral
diff = np.zeros(T_raw.size - 1)
for i in range(T_raw.size - 1):
    diff[i] = abs(T_raw[i+1] - T_raw[i])
    
H = ufloat(np.mean(diff), np.std(diff))

integral = -integrate.simps(I_raw, T_raw) / (I_raw * unp.nominal_values(H))
ln_integral = np.log(np.abs(integral))
T_inv = 1/T_raw
# linear fit
p_int, cov_int = np.polyfit(T_inv, ln_integral, deg=1, cov=True)
err_int = np.sqrt(np.diag(cov_int))
# calculate W
W_2 = ufloat(p_int[0],err_int[0])*k_ev
# print results
print('Stromdichtenansatz linearer Fit:')
print(f'm = {ufloat(p_int[0],err_int[0]):.3uS}')
print(f'b = {ufloat(p_int[1],err_int[1]):.3uS}')
print(f'W = m*k_B = {W_2:.16uS} [eV]')
print('\n')
#plot
X = np.linspace(T_inv[0], T_inv[-1], 100)
fig, ax = plt.subplots()
ax.plot(x, p_int[0]*x + p_int[1], color='mediumblue', label='linearer Fit')
ax.plot(T_inv, ln_integral,'+', color='royalblue', label = 'Integral')
ax.set_xlabel(r'$1/T/1/\si{\kelvin}$')
ax.set_ylabel(r"$\ln\left(\frac{\int_{T_0}^{T_{|I=0}} I(T')dT'}{b I(T)}\right)$")
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig('build/plot2_2.pdf')
fig.clf()

### relaxation time
# characteristic
tau_1 = k_ev*T_max**2/b/W_1*unp.exp(-W_1/k_ev/T_max)
tau_2 = unp.exp(ufloat(p_int[1],err_int[1]))
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
fig.savefig('build/plot2_t.pdf')
fig.clf()

### literature
W_lit = ufloat(0.66, 0.01)
dW = np.abs((W_1-W_lit)/W_lit)
print(dW)
