import numpy as np
import matplotlib.pyplot as plt

def parratt(a_i, delta_1, delta_2, sigma_1, sigma_2, d_2, b_1, b_2):
    n_2 = 1.0 - delta_1 +b_1
    n_3 = 1.0 - delta_2 + b_2
    a_i = np.deg2rad(a_i)
    k = 2 * np.pi / wl
    kd_1 = k * np.sqrt(n_1 ** 2 - np.cos(a_i, dtype = np.complex) ** 2)
    kd_2 = k * np.sqrt(n_2 ** 2 - np.cos(a_i, dtype = np.complex) ** 2)
    kd_3 = k * np.sqrt(n_3 ** 2 - np.cos(a_i, dtype = np.complex) ** 2)
    
    r_12 = (kd_1 - kd_2) / (kd_1 + kd_2) * np.exp(-2 * kd_1 * kd_2 * sigma_1 ** 2)
    r_23 = (kd_2 - kd_3) / (kd_2 + kd_3) * np.exp(-2 * kd_2 * kd_3 * sigma_2 ** 2)

    x_2 = np.exp(-2j * kd_2 * d_2) * r_23
    x_1 = (r_12 + x_2) / (1 + r_12 * x_2)
    par = np.abs(x_1)**2
    return par

#Import Messwerte
ref_x, ref_y = np.genfromtxt('Messwerte/omega_tet.txt', unpack=True)
diff_x, diff_y = np.genfromtxt('Messwerte/diffusor.txt' , unpack=True)

#Reflektivität
I_0 = 1114508.0792544584
R_ref = ref_y / (5 * I_0)
R_diff = diff_y / (5 * I_0)
R= R_ref - R_diff
#Geometriefaktor
a_g = 0.4
R_g = np.zeros(np.size(R))

for i in np.arange(np.size(ref_x)):
    if(ref_x[i] <= a_g and ref_x[i] > 0 ):
        R_g[i] = R[i] * np.sin(np.deg2rad(a_g)) / np.sin(np.deg2rad(ref_x[i]))
    else:
        R_g[i] = R[i]

#Parratt Parameter
n_1 = 1.0               #Brechungsindex Luft
d_1 = 0.0               #Schichtdicke Luft
d_2 = 8.6 * 10 ** (-8) #Schichtdicke der Porbe      
wl = 1.54e-10           #Wellenlänge

delta_1 = 0.3 * 10 ** (-6)
delta_2 = 6.3 * 10 ** (-6)
sigma_1 = 1.0 * 10 ** (-10) 
sigma_2 = 5.5 *10 ** (-10)
b_1 = (delta_1 / 200) # Aus L. G. Parratt. „Surface Studies of Solids by Total Reflection of X-Rays“.
b_2 = (delta_2/ 40) 


print(delta_1, delta_2, sigma_1, sigma_2, d_2, b_1, b_2)
par = parratt(ref_x, delta_1, delta_2, sigma_1, sigma_2, d_2, b_1, b_2)


#Kritischer Winkel
a_Poly = np.rad2deg(np.sqrt(2 * delta_1))
a_Si = np.rad2deg(np.sqrt(2 * delta_2))
print('Kritischer Winkel Polys:', a_Poly)
print('Kritischer Winkel Silicium: ', a_Si)

#Plot
plt.plot(ref_x, R_g, '-', label = 'Reflektivität mit Geometriefaktor')
plt.plot(ref_x, par, '-', label = 'Parratt-Kurve')
plt.xlabel('\u03B1 / °')
plt.ylabel('R')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('Graphen/Parrat_Algorthmus.pdf')