# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:17:32 2023

Used for the purpose of extracting permittivity results from measured sample
data. Measured data of MUT and air and metal to collect the proper S-Parameter (SP)
matrix. Proper SP matrix is then fed through a Nicholson-Ross-Weir method to determine
the complex permittivity.

Data Input:
    L_Matrix = SP matrix of airline with only air inside.
    T_Matrix = SP matrix of airline with the actual MUT inside.
    M_Matrix = SP matrix of airline with identical sample size but made of metal.
    
    d = length of sample
    f = frequency of given point

Airline Calibration Equations:
    S11_MUT = -1 * G[T11 - L11] / G[M11 - L11]
    
    If metal sample is unavailable, the calculation can be ran without
    but the accuracy will be slightly lowered, equation then becomes
    
    S11_MUT = G[T11 - L11]

    S21_MUT = G[T21] / G[L21]
    
    Where G[x] is a timegate that is performed by using an inverse fourier
    transform to convert to time domain and take only the first set of reflections.
    
    Once S11 and S21 have been calculated with airline calibration, they
    are run through Nicholson-Ross-Weir method.

Nicholson-Ross-Weir Equations:
    
    X = (S11^2 - S21^2 + 1) / (2 * S11)
    
    Gamma = X +- sqrt(X^2 - 1)
    
    Gamma value has a plus or minus, the value that is actually used is the
    one that has a magnitude(Gamma) < 1
    
    T = (S11 + S21 - Gamma) / (1-(S11+S21)*Gamma)
    
    1 / V^2 = -(1/(2*pi*d) * ln (1/T))^2
    
    Permeability Mu_r is then calculated to be,
    
    Mu_r = (1 + Gamma) / (V * (1 - Gamma) * sqrt((1/w_0)^2 - (1/w_c)^2))
    
    Permittivity eps_r is then calculated to be,
    
    eps_r = (w_0^2/Mu_r) * ((1/w_c)^2 + 1/V^2)
    
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
import csv
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

# Creation of window for tkinter file dialogs
root = tk.Tk()
root.attributes('-alpha', 0.0)
root.attributes('-topmost', True)
root.withdraw()


'''
Conversion of S2P file of measured S-parameter data into matrices
that can be used for data post processing
'''
freq = []
s11_real = []
s11_im = []
s21_real = []
s21_im = []
s12_real = []
s12_im = []
s22_real = []
s22_im = []

# Selection of empty airline s2p file
filepath = filedialog.askopenfilename(parent = root,title='Select Empty Airline S2P', filetypes=[("S-Parameter File",'*.s2p')])
file = open(filepath, "r")

# S2P file parsing
for line in file:
    curr_line = line.split()
    if curr_line[0].replace('.', '', 1).isdigit():
        freq.append(float(curr_line[0]))
        s11_real.append(float(curr_line[1]))
        s11_im.append(float(curr_line[2]))
        s21_real.append(float(curr_line[3]))
        s21_im.append(float(curr_line[4]))
        s12_real.append(float(curr_line[5]))
        s12_im.append(float(curr_line[6]))
        s22_real.append(float(curr_line[7]))
        s22_im.append(float(curr_line[8]))
        
file.close()

# Filename for resulting CSV, which is passed into the program
savepath_empty = filedialog.asksaveasfilename(defaultextension='.csv')
with open(savepath_empty, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['Frequency', 'S11', 'S21', 'S12', 'S22'])
    
    for idx,line in enumerate(freq):
        writer.writerow([freq[idx],
                         str(s11_real[idx] + (s11_im[idx]*1j)), 
                         str(s21_real[idx] + (s21_im[idx]*1j)),
                         str(s12_real[idx] + (s12_im[idx]*1j)),
                         str(s22_real[idx] + (s22_im[idx]*1j))])
csvfile.close()

''' 
Repeat code for the material s2p file
'''
freq = []
s11_real = []
s11_im = []
s21_real = []
s21_im = []
s12_real = []
s12_im = []
s22_real = []
s22_im = []

# Selection of empty airline s2p file
filepath = filedialog.askopenfilename(title='Select MUT Airline S2P', filetypes=[("S-Parameter File",'*.s2p')])
file = open(filepath, "r")

# S2P file parsing
for line in file:
    curr_line = line.split()
    if curr_line[0].replace('.', '', 1).isdigit():
        freq.append(float(curr_line[0]))
        s11_real.append(float(curr_line[1]))
        s11_im.append(float(curr_line[2]))
        s21_real.append(float(curr_line[3]))
        s21_im.append(float(curr_line[4]))
        s12_real.append(float(curr_line[5]))
        s12_im.append(float(curr_line[6]))
        s22_real.append(float(curr_line[7]))
        s22_im.append(float(curr_line[8]))
        
file.close()

# Filename for resulting CSV, which is passed into the program
savepath_mut = filedialog.asksaveasfilename(defaultextension='.csv')
with open(savepath_mut, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['Frequency', 'S11', 'S21', 'S12', 'S22'])
    
    for idx,line in enumerate(freq):
        writer.writerow([freq[idx],
                         str(s11_real[idx] + (s11_im[idx]*1j)), 
                         str(s21_real[idx] + (s21_im[idx]*1j)),
                         str(s12_real[idx] + (s12_im[idx]*1j)),
                         str(s22_real[idx] + (s22_im[idx]*1j))])
csvfile.close()

# Load sample data as matrices
air_matrix = np.genfromtxt(
    savepath_empty, 
    delimiter=',', 
    skip_header=1,
    dtype=complex, 
    converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(5)}
)

mut_matrix = np.genfromtxt(
    savepath_mut, 
    delimiter=',', 
    skip_header=1,
    dtype=complex, 
    converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(5)}
)

# Constants to be used for the calculation
c = 299792458
sample_length = simpledialog.askfloat("Sample Length", "Please enter the length of the sample in mm:")
cutoff_frequency = simpledialog.askfloat("Cutoff Frequency", "Please enter the cutoff frequency in GHz:")
cutoff_wavelength = c / (cutoff_frequency * pow(10,9))
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
wavelength_term = 1 / (np.sqrt(np.power(1/wavelength,2) - pow(1/cutoff_wavelength,2)))

permeability = np.multiply(np.multiply(wavelength_term, big_lambda), gamma_term)

epsterm1 = np.divide(np.square(wavelength), permeability)
epsterm2 = np.add(1 / pow(cutoff_wavelength, 2), lambda_interim)

eps = np.multiply(epsterm1, epsterm2)

# plot
fig, ax = plt.subplots()

#plt.style.use('_mpl-gallery')
ax.plot(np.real(air_matrix[:,0]), np.real(eps), linewidth=2.0)
#ax.plot(np.real(air_matrix[:,0]), np.imag(eps) / np.real(eps), linewidth=2.0)
plt.xlabel("Frequency (GHz)")
plt.ylabel("ε_r")
plt.title("NT_Sample2")

plt.grid()
plt.show()