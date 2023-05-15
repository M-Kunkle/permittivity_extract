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
    
    Γ = X +- sqrt(X^2 - 1)
    
    Γ value has a plus or minus, the value that is actually used is the
    one that has a magnitude(Γ) < 1
    
    T = (S11 + S21 - Γ) / (1-(S11+S21)*Γ)
    
    1 / Λ ^2 = -(1/(2*pi*d) * ln (1/T))^2
    
    Permeability, μ_r, is then calculated to be,
    
    μ_r = (1 + Γ) / (Λ * (1 - Γ) * sqrt((1/ω_0)^2 - (1/ω_c)^2))
    
    Permittivity ε_r is then calculated to be,
    
    ε_r = (ω_0^2 / μ_r) * ((1 / ω_c)^2 + 1 / Λ^2)
    
References:
    1. A. M. Hassan, J. Obrzut and E. J. Garboczi, "A  Q  -Band Free-Space Characterization of 
    Carbon Nanotube Composites," in IEEE Transactions on Microwave Theory and Techniques, 
    vol. 64, no. 11, pp. 3807-3819, Nov. 2016, doi: 10.1109/TMTT.2016.2603500.
    
    2. A. M. Nicolson and G. F. Ross, "Measurement of the Intrinsic Properties of Materials by
    Time-Domain Techniques," in IEEE Transactions on Instrumentation and Measurement,
    vol. 19, no. 4, pp. 377-382, Nov. 1970, doi: 10.1109/TIM.1970.4313932.
        
        
@author: M. Kunkle
"""

import tkinter
import matplotlib as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import math
import csv
import numpy as np

plt.style.use('bmh')



class GraphFrame(tkinter.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()


root = tkinter.Tk()
root.wm_title("Material Characterization")
root.geometry("+100+20")

np.seterr(divide='ignore', invalid='ignore')

air_cal_dialog = tkinter.messagebox.askyesno(title="Airline Calibration", message="Do you require an airline calibration?")
data_format = tkinter.messagebox.askyesno(title="Data Format Selection", message="Is your data formatted into an s2p file?")
ghz_dialog = tkinter.messagebox.askyesno(title="Frequency Selection", message="Frequency in GHz?")

if(data_format):
    ffx = tkinter.messagebox.askyesno(title="Fieldfox S2P?", message="S2P from fieldfox?")
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
        filepath = tkinter.filedialog.askopenfilename(parent = root,title='Select Empty Airline S2P', filetypes=[("S-Parameter File",'*.s2p')])
        file = open(filepath, "r")
        
        if(ffx):
            # ffx S2P file parsing
            for line in file:
                curr_line = line.split()
                if curr_line[0].replace('.', '', 1).isdigit():
                    freq.append(float(curr_line[0]))
                    s11_real.append(10**(float(curr_line[1])/20)*math.cos((math.pi*float(curr_line[2]))/180))
                    s11_im.append(10**(float(curr_line[1])/20)*math.sin((math.pi*float(curr_line[2]))/180))
                    s21_real.append(10**(float(curr_line[3])/20)*math.cos((math.pi*float(curr_line[4]))/180))
                    s21_im.append(10**(float(curr_line[3])/20)*math.sin((math.pi*float(curr_line[4]))/180))
                    s12_real.append(10**(float(curr_line[5])/20)*math.cos((math.pi*float(curr_line[6]))/180))
                    s12_im.append(10**(float(curr_line[5])/20)*math.sin((math.pi*float(curr_line[6]))/180))
                    s22_real.append(10**(float(curr_line[7])/20)*math.cos((math.pi*float(curr_line[8]))/180))
                    s22_im.append(10**(float(curr_line[7])/20)*math.sin((math.pi*float(curr_line[8]))/180))
            file.close()
            
        else:
            # S2P file parsing
            for line in file:
                curr_line = line.split()
                if curr_line[0].replace('.', '', 1).isdigit():
                    freq.append(float(curr_line[0]))
                    s11_real.append(float(curr_line[1])*math.cos((math.pi*float(curr_line[2]))/180))
                    s11_im.append(float(curr_line[1])*math.sin((math.pi*float(curr_line[2]))/180))
                    s21_real.append(float(curr_line[3])*math.cos((math.pi*float(curr_line[4]))/180))
                    s21_im.append(float(curr_line[3])*math.sin((math.pi*float(curr_line[4]))/180))
                    s12_real.append(float(curr_line[5])*math.cos((math.pi*float(curr_line[6]))/180))
                    s12_im.append(float(curr_line[5])*math.sin((math.pi*float(curr_line[6]))/180))
                    s22_real.append(float(curr_line[7])*math.cos((math.pi*float(curr_line[8]))/180))
                    s22_im.append(float(curr_line[7])*math.sin((math.pi*float(curr_line[8]))/180))
            file.close()
        
        # Filename for resulting CSV, which is passed into the program
        savepath_empty = tkinter.filedialog.asksaveasfilename(defaultextension='.csv')
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
    filepath = tkinter.filedialog.askopenfilename(title='Select MUT Airline S2P', filetypes=[("S-Parameter File",'*.s2p')])
    file = open(filepath, "r")
    
    # S2P file parsing
    if(ffx):
        # ffx S2P file parsing
        for line in file:
            curr_line = line.split()
            if curr_line[0].replace('.', '', 1).isdigit():
                freq.append(float(curr_line[0]))
                s11_real.append(10**(float(curr_line[1])/20)*math.cos((math.pi*float(curr_line[2]))/180))
                s11_im.append(10**(float(curr_line[1])/20)*math.sin((math.pi*float(curr_line[2]))/180))
                s21_real.append(10**(float(curr_line[3])/20)*math.cos((math.pi*float(curr_line[4]))/180))
                s21_im.append(10**(float(curr_line[3])/20)*math.sin((math.pi*float(curr_line[4]))/180))
                s12_real.append(10**(float(curr_line[5])/20)*math.cos((math.pi*float(curr_line[6]))/180))
                s12_im.append(10**(float(curr_line[5])/20)*math.sin((math.pi*float(curr_line[6]))/180))
                s22_real.append(10**(float(curr_line[7])/20)*math.cos((math.pi*float(curr_line[8]))/180))
                s22_im.append(10**(float(curr_line[7])/20)*math.sin((math.pi*float(curr_line[8]))/180))
        
    else:
        # S2P file parsing
        for line in file:
            curr_line = line.split()
            if curr_line[0].replace('.', '', 1).isdigit():
                freq.append(float(curr_line[0]))
                s11_real.append(float(curr_line[1])*math.cos((math.pi*float(curr_line[2]))/180))
                s11_im.append(float(curr_line[1])*math.sin((math.pi*float(curr_line[2]))/180))
                s21_real.append(float(curr_line[3])*math.cos((math.pi*float(curr_line[4]))/180))
                s21_im.append(float(curr_line[3])*math.sin((math.pi*float(curr_line[4]))/180))
                s12_real.append(float(curr_line[5])*math.cos((math.pi*float(curr_line[6]))/180))
                s12_im.append(float(curr_line[5])*math.sin((math.pi*float(curr_line[6]))/180))
                s22_real.append(float(curr_line[7])*math.cos((math.pi*float(curr_line[8]))/180))
                s22_im.append(float(curr_line[7])*math.sin((math.pi*float(curr_line[8]))/180))
            
    file.close()
    
    # Filename for resulting CSV, which is passed into the program
    savepath_mut = tkinter.filedialog.asksaveasfilename(defaultextension='.csv')
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
        savepath_empty = tkinter.filedialog.askopenfilename(title="Select Empty Airline Data CSV", filetypes=[("Comma Separated Values", '*.csv')])
    savepath_mut = tkinter.filedialog.askopenfilename(title="Select MUT Airline Data CSV", filetypes=[("Comma Separated Values", '*.csv')])

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

# Constants to be used for the calculation
c = 299792458
sample_length = tkinter.simpledialog.askfloat("Sample Length", "Please enter the length of the sample in m:")
cutoff_frequency = tkinter.simpledialog.askfloat("Cutoff Frequency", "Please enter the cutoff frequency in GHz:")

if(cutoff_frequency != 0):
        cutoff_λ = c / (cutoff_frequency * pow(10,9))
    

if(ghz_dialog):
    freq = np.real(mut_matrix[:,0])
else:
    freq = np.real(mut_matrix[:,0])*1e9
    
λ = c / freq
beta = np.divide(2*math.pi, λ)


'''
Cycle through entirety of matrices, calculations are done elementwise at each
specific frequency point that has been measured.
'''

if(air_cal_dialog):
    '''
    Airline calibration:
    '''
    
    s11_mut = np.divide(-1 * (mut_matrix[:,1] -  air_matrix[:,1]), 1 - air_matrix[:,1])
    
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
X = (1 - np.square(s21_mut) + np.square(s11_mut)) / (2 * s11_mut)

# Γ = X +- sqrt(X^2 - 1)
Γ_1 = X + np.sqrt(np.square(X) - 1)
Γ_2 = X - np.sqrt(np.square(X) - 1)
Γ = np.where(np.abs(Γ_1) < 1, Γ_1, Γ_2)

# T = (S11 + S21 - Γ) / (1-(S11+S21)*Γ)
T = (s11_mut + s21_mut - Γ) / (1 - Γ * (s11_mut + s21_mut))

# z = 
z = np.sqrt((np.square(1 + s11_mut) - np.square(s21_mut)) / \
            (np.square(1 - s11_mut) - np.square(s21_mut)))

# delay = e^(-*d)
delay = (1 - np.square(s11_mut) + np.square(s21_mut)) / (2 * s21_mut) \
        + (2 * s11_mut) / (s21_mut * (z - (1 / z)))  

ln_delay = np.log(np.absolute(delay)) + 1j * np.angle(delay)

inv_Λ_sq = np.multiply(np.square(np.multiply(1 / (2 * math.pi * sample_length), ln_delay)), -1)
inv_Λ = np.sqrt(inv_Λ_sq)

μ = inv_Λ * ((1 + Γ) / (1 - Γ)) * λ

ε_term1 = np.divide(np.square(λ),μ)
ε = ε_term1 * inv_Λ_sq

a,b,c = np.polyfit(freq / 1e9, np.real(ε), 2)
d,e,f = np.polyfit(freq / 1e9, np.real(μ), 2)

test = np.fft.ifft(s11_mut)

eps_graph = GraphFrame(root, highlightbackground="black", highlightthickness=1)
eps_graph.ax.plot(freq / 1e9, np.real(ε), linewidth=2.0)
#eps_graph.ax.plot(freq / 1e9, a*((freq / 1e9)**2) + b*(freq / 1e9) + c, linewidth=2.0)
eps_graph.grid(column=0, row=0, padx=10, pady=4)
eps_graph.ax.set_xlabel("freq [GHz]")
eps_graph.ax.set_ylabel("eps_r")
eps_graph.ax.set_title("Relative Permittivity")
eps_graph.ax.grid(visible=True, axis='both')

lt_graph = GraphFrame(root, highlightbackground="black", highlightthickness=1)
lt_graph.ax.plot(freq / 1e9, -np.imag(ε) / np.real(ε), linewidth=2.0)
lt_graph.grid(column=1, row=0, padx=10, pady=4)
lt_graph.ax.set_xlabel("freq [GHz]")
lt_graph.ax.set_ylabel("loss tangent")
lt_graph.ax.set_title("Loss Tangent")
lt_graph.ax.grid(visible=True, axis='both')

s11_graph = GraphFrame(root, highlightbackground="black", highlightthickness=1)
s11_graph.ax.plot(freq / 1e9, 20*np.log10(np.absolute(s11_mut)), linewidth=2.0)
s11_graph.grid(column=0, row=1, padx=10, pady=4)
s11_graph.ax.set_xlabel("freq [GHz]")
s11_graph.ax.set_ylabel("S11 [dB]")
s11_graph.ax.set_title("S11")
s11_graph.ax.grid(visible=True, axis='both')

mu_graph = GraphFrame(root, highlightbackground="black", highlightthickness=1)
#mu_graph.ax.plot(np.fft.ifft(s11_mut))
mu_graph.ax.plot(freq / 1e9, np.real(μ), linewidth=2.0)
#mu_graph.ax.plot(freq / 1e9, d*((freq / 1e9)**2) + e*(freq / 1e9) + f, linewidth=2.0)
mu_graph.grid(column=1, row=1, padx=10, pady=4)
mu_graph.ax.set_xlabel("freq [GHz]")
mu_graph.ax.set_ylabel("mu")
mu_graph.ax.set_title("Relative Permeability")
mu_graph.ax.grid(visible=True, axis='both')

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy, bg="#FEB09F")
button_quit.grid(column=0, row=2, columnspan=2, sticky=(tkinter.E, tkinter.W), pady=4, padx=4)

ask_csv = tkinter.messagebox.askyesno(title="Data Export", message="Save data to an csv file?")

if(ask_csv):
    csv_file = tkinter.filedialog.asksaveasfilename(defaultextension='.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['Frequency', 'S11', 'S21', 'Complex Permittivity',\
                         'Complex Permeability', 'Loss Tangent', 'Refractive Index'])
        
        for idx,line in enumerate(freq):
            writer.writerow([freq[idx],
                             str(np.real(s11_mut[idx])) + str(np.imag(s11_mut[idx])) + 'i', 
                             str(np.real(s21_mut[idx])) + str(np.imag(s21_mut[idx])) + 'i',
                             str(np.real(ε[idx])) + str(np.imag(ε[idx])) + 'i',
                             str(np.real(μ[idx])) + str(np.imag(μ[idx])) + 'i',
                             str(-np.imag(ε[idx]) / np.real(ε[idx])),
                             str(np.sqrt(np.real(ε[idx]) * np.real(μ[idx])))
                             ])
    csvfile.close()

tkinter.mainloop()