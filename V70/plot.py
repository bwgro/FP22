import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from uncertainties import ufloat

# Drehschieber
x3 = unp.uarray([0.143, 0.507], [0.008, 0.029])
y3 = unp.uarray([0.144, 0.144], [0.014, 0.014])

x2 = unp.uarray([0.507, 2.2], [0.029, 0.13])
y2 = unp.uarray([0.37, 0.37], [0.04, 0.04]) 

x1 = unp.uarray([2.2, 1000], [0.13, 1.7])
y1 = unp.uarray([1.07, 1.07], [0.11, 0.11])

fig, ax = plt.subplots()

ax.fill_between(noms(x1), noms(y1[0])-stds(y1[0]), noms(y1[0])+stds(y1[0]), alpha=0.2)
ax.errorbar(noms(x1), noms(y1), xerr=stds(x1), yerr=stds(y1), capsize=3, label=r'$S_{E,1}$')
ax.fill_between(noms(x2), noms(y2[0])-stds(y2[0]), noms(y2[0])+stds(y2[0]), alpha=0.2)
ax.errorbar(noms(x2), noms(y2), xerr=stds(x2), yerr=stds(y2), capsize=2, label=r'$S_{E,2}$')
ax.fill_between(noms(x3), noms(y3[0])-stds(y3[0]), noms(y3[0])+stds(y3[0]), alpha=0.2)
ax.errorbar(noms(x3), noms(y3), xerr=stds(x3), yerr=stds(y3), capsize=1, label=r'$S_{E,3}$')

ax.errorbar(0.5, 0.62, yerr=0.07, label=r'$S_{L,0.5}$', fmt='+', capsize=2)
ax.errorbar(10, 1.31, yerr=0.13, label=r'$S_{L,10}$', fmt='+', capsize=3)
ax.errorbar(50, 1.17, yerr=0.12, label=r'$S_{L,50}$', fmt='+', capsize=3)
ax.errorbar(100, 1.10, yerr=0.11, label=r'$S_{L,100}$', fmt='+', capsize=3)

ax.set_xscale('symlog')
ax.legend(loc='best', fontsize='small')
ax.set_xlabel(r'p/\unit{\milli\bar}')
ax.set_ylabel(r'S/\unit{\liter\per\second}')
ax.grid()
fig.tight_layout()
fig.savefig('abb/plot1.pdf')

# Turbo
x3 = unp.uarray([3e-4, 1.64e-4], [0, 0.28e-4])
y3 = unp.uarray([11.56, 11.56], [0, 1.16])

x2 = unp.uarray([1.64e-4, 6.09e-5], [0.28e-4, 1.05e-5])
y2 = unp.uarray([3.84, 3.84], [0.38, 0.38]) 

x1 = unp.uarray([5.04e-5, 3.81e-5], [0.87e-5, 0.66e-5])
y1 = unp.uarray([0.14, 0.14], [0.01, 0.01])

fig, ax = plt.subplots()

ax.fill_between(noms(x1), noms(y1[0])-stds(y1[0]), noms(y1[0])+stds(y1[0]), alpha=0.2)
ax.errorbar(noms(x1), noms(y1), xerr=stds(x1), yerr=stds(y1), capsize=1, label=r'$S_{E,1}$')
ax.fill_between(noms(x2), noms(y2[0])-stds(y2[0]), noms(y2[0])+stds(y2[0]), alpha=0.2)
ax.errorbar(noms(x2), noms(y2), xerr=stds(x2), yerr=stds(y2), capsize=2, label=r'$S_{E,2}$')
ax.fill_between(noms(x3), noms(y3[1])-stds(y3[1]), noms(y3[1])+stds(y3[1]), alpha=0.2)
ax.errorbar(noms(x3), noms(y3), xerr=stds(x3), yerr=stds(y3), capsize=3, label=r'$S_{E,3}$')

ax.errorbar(2e-4, 8.72, yerr=1.76, label=r'$S_{L,\num{2e-4}}$', fmt='+', capsize=3)
ax.errorbar(1e-4, 5.15, yerr=1.04, label=r'$S_{L,\num{1e-4}}$', fmt='+', capsize=3)
ax.errorbar(7e-5, 4.48, yerr=0.9, label=r'$S_{L,\num{7e-5}}$', fmt='+', capsize=3)
ax.errorbar(5e-5, 3.89, yerr=0.78, label=r'$S_{L,\num{5e-5}}$', fmt='+', capsize=3)
ax.set_xlim(0,3e-4)
#ax.set_xticks([0, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5, 30e-5], [r'0', r'$\num{5e-5}$', r'$\num{10e-5}$', r'$\num{15e-5}$', r'$\num{20e-5}$', r'$\num{25e-5}$', r'$\num{30e-5}$'])
ax.legend(loc='best', fontsize='small')
ax.set_xlabel(r'p/\unit{\milli\bar}')
ax.set_ylabel(r'S/\unit{\liter\per\second}')
ax.set_xscale('symlog')
ax.set_xticks([0, 0.00010, 0.00020, 0.00030])
ax.grid()
fig.tight_layout()
fig.savefig('abb/plot2.pdf')