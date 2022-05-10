import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from uncertainties import ufloat


# Evakuierungskurve
t, dt, p1, p2, p3, dp = np.genfromtxt('data/turbo_evac.csv', unpack=True)

# Enddruck
pE = ufloat(6.7e-3, 5e-5)

T = unp.uarray(t,dt)
p_1 = unp.uarray(p1,dp)
p_2 = unp.uarray(p2,dp)
p_3 = unp.uarray(p3,dp)
P= np.mean([p_1,p_2,p_3], axis=0)

# Messdaten zu Latex-Tabelle
with open('content/turbo_evac.tex', 'w') as f:
    for i in range(t.size):
        f.write(f'{T[i]:S} && {p_1[i]:S} && {p_2[i]:S} && {p_3[i]:S} && {P[i]:S} \\\\ \n')

# Anpassen an Enddruck
p_1 = p_1 - pE
p_2 = p_2 - pE
p_3 = p_3 - pE
P = P - pE

# Lograthmischer Ausdruck nach Enddruckanpassung
p_1 = unp.log(p_1/p_1[0])
p_2 = unp.log(p_2/p_2[0])
p_3 = unp.log(p_3/p_3[0])
#P = unp.log(P/P[0])

# Ausgleichsgeraden in Abschnitten
print('Parameter der Ausgleichsgeraden zur Evakuierungskurve der Turbopumpe:')
params1 = np.polyfit(t[0:2], noms(P[0:2]), deg=1)
#uParams1 = unp.uarray(params1, np.sqrt(np.diag(cov_matrix1)))
print(f'Abschnitt 1: m = {params1[0]}, b = {params1[1]}')
params2,cov_matrix2 = np.polyfit(t, noms(P), deg=1, cov=True)
uParams2 = unp.uarray(params2, np.sqrt(np.diag(cov_matrix2)))
print(f'Abschnitt 2: m = {uParams2[0]:S}, b = {uParams2[1]:S}')
params3,cov_matrix3 = np.polyfit(t, noms(P), deg=1, cov=True)
uParams3 = unp.uarray(params3, np.sqrt(np.diag(cov_matrix3)))
print(f'Abschnitt 3: m = {uParams3[0]:S}, b = {uParams3[1]:S}\n')

# Plot
x = np.linspace(0,120,1200)
fig, ax = plt.subplots()
ax.errorbar(t, noms(P), xerr=0.2, yerr=stds(P), fmt='k+', ecolor='r', label=r'Messwerte mit Fehlerbalken')
ax.plot(x[0:110], params1[0]*x[0:110] + params1[1], label=r'Abschnitt 1')
#ax.plot(x, params2[0]*x + params2[1], label=r'Abschnitt 2')
#ax.plot(x, params3[0]*x + params3[1], label=r'Abschnitt 3')
ax.set_ylabel(r'$\ln\left(\frac{\bar{p}(t)-p_E}{\bar{p}_0-p_E}\right)$')
ax.set_xlabel(r'$t/\si{\second}$')
ax.set_title('Evakuierungskurve der Turbopumpe')
ax.grid()
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('abb/turbo_evac.pdf')