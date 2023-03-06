# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:42:50 2023

@author: mkrxp
"""

import csv
import tkinter as tk
from tkinter import filedialog

freq = []
s11_real = []
s11_im = []
s21_real = []
s21_im =[]

def main():
    
    root = tk.Tk()
    root.withdraw()
    
    filepath = filedialog.askopenfilename()
    
    file = open(filepath, "r")
    
    
    for line in file:
        curr_line = line.split()
        if curr_line[0].replace('.', '', 1).isdigit():
            freq.append(float(curr_line[0]))
            s11_real.append(float(curr_line[1]))
            s11_im.append(float(curr_line[2]))
            s21_real.append(float(curr_line[3]))
            s21_im.append(float(curr_line[4]))
            
    file.close()
            
    savepath = filedialog.asksaveasfilename(defaultextension='.csv')
    with open(savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['Frequency', 'S11 Re', 'S11 Im', 'S21 Re', 'S21 Im'])
        
        for idx,line in enumerate(freq):
            writer.writerow([freq[idx], s11_real[idx], s11_im[idx], s21_real[idx], s21_im[idx]])
        
    csvfile.close()
    
    savepath = filedialog.asksaveasfilename(defaultextension='.csv')
    with open(savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['Frequency', 'S11', 'S21'])
        
        for idx,line in enumerate(freq):
            writer.writerow([freq[idx], str(s11_real[idx] + (s11_im[idx] * 1j)), str(s11_real[idx] + (s11_im[idx]*1j))])
        
    csvfile.close()
    
main()