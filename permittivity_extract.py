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
    
    Test setup requires a material machined into a hollow cylinder with inner
    dimension of the cylinder being the size of the center conductor pin. The 
    outside cylinder radius should fill the airline as much as possible. The 
    length of the sample is ideally the full length of the airline, but if the 
    sample does not fill the airline, it must be completely as centered inside
    the airline, with equal amounts of air on either side.
    
    S11_MUT = -1 * G[T11 - L11] / G[M11 - L11]
    
    If metal sample is unavailable, the calculation can be ran without
    but the accuracy will be slightly lowered, equation then becomes
    
    S11_MUT = G[T11 - L11]

    S21_MUT = (G[T21] / G[L21]) * e^(-jBd)
    
    The exponential term on the end of the S21 normalization is used to account
    for the phase delay that occurs when the wave is passed through the dielectric.
    
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
    
    2. A. M. Nicolson and G. F. Ross, "Measurement of the Intrinsic Properties of Materials by
    Time-Domain Techniques," in IEEE Transactions on Instrumentation and Measurement,
    vol. 19, no. 4, pp. 377-382, Nov. 1970, doi: 10.1109/TIM.1970.4313932.
        
        
@author: M. Kunkle
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox

# Creation of window for tkinter file dialogs
root = tk.Tk()
root.attributes('-alpha', 0.0)
root.attributes('-topmost', True)
root.withdraw()

np.seterr(divide='ignore', invalid='ignore')

air_cal_dialog = messagebox.askyesno(title="Airline Calibration", message="Do you require an airline calibration?")
data_format = messagebox.askyesno(title="Data Format Selection", message="Is your data formatted into an s2p file?")
gate_dialog = messagebox.askyesno(title="Time Gating Selection", message="Apply time gating to S-Parameters?")

if(data_format):
    '''
    Conversion of S2P file of measured S-parameter data into matrices
    that can be used for data post processing
    '''
    if(air_cal_dialog):
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
                s11_real.append(float(curr_line[1])*np.cos(float(curr_line[2])))
                s11_im.append(float(curr_line[1])*np.sin(float(curr_line[2])))
                s21_real.append(float(curr_line[3])*np.cos(float(curr_line[4])))
                s21_im.append(float(curr_line[3])*np.sin(float(curr_line[4])))
                s12_real.append(float(curr_line[5])*np.cos(float(curr_line[6])))
                s12_im.append(float(curr_line[5])*np.sin(float(curr_line[6])))
                s22_real.append(float(curr_line[7])*np.cos(float(curr_line[8])))
                s22_im.append(float(curr_line[7])*np.cos(float(curr_line[8])))
                
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
            s11_real.append(float(curr_line[1])*np.cos(float(curr_line[2])))
            s11_im.append(float(curr_line[1])*np.sin(float(curr_line[2])))
            s21_real.append(float(curr_line[3])*np.cos(float(curr_line[4])))
            s21_im.append(float(curr_line[3])*np.sin(float(curr_line[4])))
            s12_real.append(float(curr_line[5])*np.cos(float(curr_line[6])))
            s12_im.append(float(curr_line[5])*np.sin(float(curr_line[6])))
            s22_real.append(float(curr_line[7])*np.cos(float(curr_line[8])))
            s22_im.append(float(curr_line[7])*np.cos(float(curr_line[8])))
            
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
    
else:
    if(air_cal_dialog):
        savepath_empty = filedialog.askopenfilename(title="Select Empty Airline Data CSV", filetypes=[("Comma Separated Values", '*.csv')])
    savepath_mut = filedialog.askopenfilename(title="Select MUT Airline Data CSV", filetypes=[("Comma Separated Values", '*.csv')])

if(air_cal_dialog):
    # Load sample data as matrices
    air_matrix = np.genfromtxt(
        savepath_empty, 
        delimiter=',', 
        skip_header=1,
        dtype=complex, 
        converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(3)}
    )

mut_matrix = np.genfromtxt(
    savepath_mut, 
    delimiter=',', 
    skip_header=1,
    dtype=complex, 
    converters={k: lambda x: complex(x.replace(b' ', b'').decode()) for k in range(3)}
)

if(gate_dialog):
    # plot
    fig, ax = plt.subplots()
    ax.plot(np.real(mut_matrix[:,0]), 20*np.log10(abs(mut_matrix[:,1])), linewidth=2.0)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11")
    plt.grid(visible=True, axis='both')
    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(), np.fft.ifft(mut_matrix[:,1]), linewidth=2.0)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11")
    plt.grid(visible=True, axis='both')
    plt.show()

# Constants to be used for the calculation
c = 299792458
sample_length = simpledialog.askfloat("Sample Length", "Please enter the length of the sample in m:")
#cutoff_frequency = simpledialog.askfloat("Cutoff Frequency", "Please enter the cutoff frequency in GHz:")
#cutoff_wavelength = c / (cutoff_frequency * pow(10,9))
freq = np.real(mut_matrix[:,0])*1e9
wavelength = c / freq

beta = np.divide(2*math.pi, wavelength)

'''
Cycle through entirety of matrices, calculations are done elementwise at each
specific frequency point that has been measured.
'''

if(air_cal_dialog):
    '''
    Airline calibration:
    '''
    
    s11_mut = np.subtract(mut_matrix[:,1], air_matrix[:,1])
    #s11_denom = np.subtract(metal_matrix[1], air_matrix[1])
    #s11_mut = -1 * np.divide(s11_numer[1], s11_denom[1])
    
    # normalization factor split up into two separate terms
    s21_normalize = np.divide(mut_matrix[:,2], air_matrix[:,2])
    delay_term = np.exp(-1j*sample_length*beta)
    s21_mut = np.multiply(s21_normalize, delay_term)
else: 
    s11_mut = mut_matrix[:,1]
    s21_mut = mut_matrix[:,2]

'''
Nicholson-Ross-Weir Calculations
'''

# X = (S11^2 - S21^2 + 1) / (2 * S11)
X_numer = 1 - (np.square(s21_mut) - np.square(s11_mut))
X = np.divide(X_numer, np.multiply(2, s11_mut))

# Gamma = X +- sqrt(X^2 - 1)
gamma1 = np.abs(np.add(X, np.sqrt(np.subtract(np.square(X),1))))
gamma2 = np.abs(np.subtract(X, np.sqrt(np.subtract(np.square(X),1))))
gamma = np.where(gamma1 < 1, np.add(X, np.sqrt(np.subtract(np.square(X),1))), np.subtract(X, np.sqrt(np.subtract(np.square(X),1))))


# T = (S11 + S21 - Gamma) / (1-(S11+S21)*Gamma)
T_numer = s11_mut + s21_mut - gamma
T_denom = 1 - np.multiply(gamma,(s11_mut + s21_mut))
T = np.divide(T_numer, T_denom)

z_top = np.subtract(np.square(np.add(1, s11_mut)), np.square(s21_mut))
z_bottom = np.subtract(np.square(np.subtract(1, s11_mut)), np.square(s21_mut))
z = np.sqrt(np.divide(z_top, z_bottom))

delay_term1_top = np.add(np.subtract(1, np.square(s11_mut)), np.square(s21_mut))
delay_term1_bot = np.multiply(2, s21_mut)
delay_term1 = np.divide(delay_term1_top, delay_term1_bot)
delay_term2_top = np.multiply(2,s11_mut)
delay_term2_bot = np.multiply(s21_mut, np.subtract(z, np.divide(1,z)))
delay_term2 = np.divide(delay_term2_top, delay_term2_bot)
delay_term = np.add(delay_term1, delay_term2)                         

ln_delay_term1 = np.log(np.absolute(delay_term))
ln_delay_term2 = np.multiply(1j, np.angle(delay_term))
ln_delay = np.add(ln_delay_term1, ln_delay_term2)


eps_inside = np.add(np.multiply(np.divide(c,np.multiply(freq*2*3.14159,sample_length)), ln_delay), ln_delay_term2)
eps = -np.square(eps_inside)

# plot
fig, ax = plt.subplots()
ax.plot(freq, np.real(eps), linewidth=2.0)
#ax.plot(np.real(mut_matrix[:,0]), np.imag(eps) / np.real(eps), linewidth=2.0)
plt.xlabel("Frequency (GHz)")
plt.ylabel("ε_r")
plt.grid(visible=True, axis='both')
plt.show()
