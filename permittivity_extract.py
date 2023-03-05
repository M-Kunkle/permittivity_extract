# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:17:32 2023

Used for the purpose of extracting permittivity results from measured sample
data. Measured data of MUT and air and metal to collect the proper S-Parameter (SP)
matrix. Proper SP matrix is then fed through a Nicholson-Ross method to determine
the complex permittivity.

Data Input:
    L_Matrix = SP matrix of airline with only air inside.
    T_Matrix = SP matrix of airline with the actual MUT inside.
    M_Matrix = SP matrix of airline with identical sample size but made of metal.

Calibration Equations:
    S11_MUT = -1 * G[T11 - L11] / G[M11 - L11]
    S21_MUT = G[T21] / G[L21]

References:
    1. A. M. Hassan, J. Obrzut and E. J. Garboczi, "A  Q  -Band Free-Space Characterization of 
    Carbon Nanotube Composites," in IEEE Transactions on Microwave Theory and Techniques, 
    vol. 64, no. 11, pp. 3807-3819, Nov. 2016, doi: 10.1109/TMTT.2016.2603500.
    
    2. Nicholson Ross reference
        
        
@author: M. Kunkle
"""

import numpy as np
import math
import matplotlib.pyplot as plt

sample_length = .00709
# Load sample data as matrices


air_matrix = np.genfromtxt(
    'empty.csv', 
    delimiter=',', 
    skip_header=1,
    dtype=complex, 
    converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(5)}
)
#metal_matrix = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1)
mut_matrix = np.genfromtxt(
    'NT_Sample2_7.09mm,8-4,0.csv', 
    delimiter=',', 
    skip_header=1,
    dtype=complex, 
    converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(5)}
)

cutoff_wavelength = .37474
c = 299792458
wavelength = c / (air_matrix[:,0] * pow(10,9))
beta = np.divide(2*math.pi, wavelength)

# Cycle through entirety of matrices, calculations are done elementwise at each
# specific frequency point that has been measured.

s11_mut = np.subtract(mut_matrix[:,1], air_matrix[:,1])
#s11_denom = np.subtract(metal_matrix[1], air_matrix[1])
#s11_mut = -1 * np.divide(s11_numer[1], s11_denom[1])

s21_normalize = np.divide(mut_matrix[:,2], air_matrix[:,2])
delay_term = np.exp(-1j*sample_length*beta)
s21_mut = np.multiply(s21_normalize, delay_term)

X_numer = 1 - (np.square(s21_mut) - np.square(s11_mut))
X = np.divide(X_numer, np.multiply(2, s11_mut))


gamma1 = np.abs(np.add(X, np.sqrt(np.subtract(np.square(X),1))))
gamma2 = np.abs(np.subtract(X, np.sqrt(np.subtract(np.square(X),1))))

gamma = np.where(gamma1 < 1, np.add(X, np.sqrt(np.subtract(np.square(X),1))), np.subtract(X, np.sqrt(np.subtract(np.square(X),1))))

T_numer = s11_mut + s21_mut - gamma
T_denom = 1 - np.multiply(gamma,(s11_mut + s21_mut))
T = np.divide(T_numer, T_denom)

lambda_term = -1 * np.multiply(1 / (2*math.pi*sample_length), np.log(np.divide(1,T)))
lambda_interim = -1 * np.power(lambda_term, 2)
big_lambda = np.sqrt(lambda_interim)

gamma_term = np.divide(1 + gamma, 1 - gamma)
permeability = np.multiply(np.multiply(wavelength, big_lambda), gamma_term)

epsterm1 = np.divide(np.square(wavelength), permeability)
epsterm2 = np.subtract(1 / pow(cutoff_wavelength, 2), lambda_interim)

eps = np.multiply(epsterm1, epsterm2)

# plot
fig, ax = plt.subplots()

#plt.style.use('_mpl-gallery')
ax.plot(np.real(air_matrix[:,0]), np.real(eps), linewidth=2.0)
ax.plot(np.real(air_matrix[:,0]), s11_mut, linewidth=2.0)
ax.plot(np.real(air_matrix[:,0]), s21_mut, linewidth=2.0)
plt.show()


