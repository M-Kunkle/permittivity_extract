# permittivity_extract

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