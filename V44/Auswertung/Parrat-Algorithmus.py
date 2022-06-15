import numpy as np
import matplotlib.pyplot as plt

ref_x, ref_y = np.genfromtxt('Messwerte/omega_tet.txt', unpack=True)
diff_x, diff_y = np.genfromtxt('Messwerte/diffusor.txt' , unpack=True)

#Reflektivität
I_0 = 100e06
R_ref = ref_y / (5 * I_0)
R_diff = diff_y / (5 * I_0)
R= R_ref - R_diff

#Geometriefaktor
#a_g = 0.68758564108561
#R_g = np.zeros(np.size(R))

#for i in np.arange(np.size(a)):
#    if(a[i] <= a_g and a[i] > 0 ):
#        R_g[i] = R[i] * np.sin(np.deg2rad(a_g)) / np.sin(np.deg2rad(a[i]))
#    else:
#        R_g[i] = R[i]


#Parratt

n_1 = 1.0               #Brechungsindex Luft
d_1 = 0.0               #Schichtdicke Luft
d_2 = 8.6 * 10 ** (-10) #Schichtdicke der Porbe
wl = 1.54e-10           #Wellenlänge
delta_1 = 0.6 * 10 ** (-6)
delta_2 = 6.0 * 10 ** (-6)
sigma_1 = 5.5 * 10 ** (-10) #Rauigkeiten
sigma_2 = 6.45 *10 ** (-10)
b_2 = (delta_1 / 40) * 1j
b_3 = (delta_2/ 200) * 1j



def parratt(a_i, delta_1, delta_2, sigma_1, sigma_2, d_2, b_1, b_2):
    n_2 = 1.0 - delta_1 +b_2
    n_3 = 1.0 - delta_2 + b_3
    a_i = np.deg2rad(a_i)
    k = 2 * np.pi / wl
    kd_1 = k * np.sqrt(n_1 ** 2 - np.cos(a_i) ** 2)
    kd_2 = k * np.sqrt(n_2 ** 2 - np.cos(a_i) ** 2)
    kd_3 = k * np.sqrt(n_3 ** 2 - np.cos(a_i) ** 2)
    
    r_12 = (kd_1 - kd_2) / (kd_1 + kd_2) * np.exp(-2 * kd_1 * kd_2 * sigma_1 ** 2)
    r_23 = (kd_2 - kd_3) / (kd_2 + kd_3) * np.exp(-2 * kd_2 * kd_3 * sigma_2 ** 2)

    x_2 = np.exp(-2j * kd_2 * d_2) * r_23
    x_1 = (r_12 + x_2) / (1 + r_12 * x_2)

    return np.abs(x_1) **2


print(delta_1, delta_2, sigma_1, sigma_2, d_2, b_2, b_3)
par = parratt(ref_x, delta_1, delta_2, sigma_1, sigma_2, d_2, b_2, b_3)


#Kritischer Winkel
a_Poly = np.rad2deg(np.sqrt(2 * delta_1))
a_Si = np.rad2deg(np.sqrt(2 * delta_2))
print('Kritischer Winkel Polys:', a_Poly)
print('Kritischer Winkel Silicium: ', a_Si)

#Plot
plt.plot(ref_x, ref_y, '-', label = 'Messdaten')
plt.plot(ref_x, par, '-', label = 'Parratt-Kurve')
plt.xlabel('\u03B1 / °')
plt.ylabel('R')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('Graphen/Parrat_Algorthmus.pdf')