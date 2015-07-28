"""This script is used to calculate the vibrational frequency, desorption rate, 
desorption time, and freezing temperature of a given volatile"""


import numpy as np
from utils.constants import mp, kb
from scipy.optimize import brentq

def vib_freq(Ex, mx):
    
    """
    Input
    -----
    Ex: 
        binding energy in K
    mx:
        molecular weight in proton masses
        
    Output
    ------
    nu:
        vibrational frequency in s^-1
    """
    
    return 1.6 * 1e11 * np.sqrt(Ex / mx)

def Rdes(mx, Ex, Tx):
    
    return vib_freq(Ex, mx) * np.exp(- Ex / Tx) 
    
def tevap(mx, Ex, Tx):
    
    return 1 / Rdes(mx, Ex, Tx)
    
def v_thermal(T, mu):
    
    """
    Input
    -----
    T:
        temperature in K
    mu:
        molecular weight in proton masses
        
    Output
    ------
    v_thermal:
        thermal velocity in cm s^-1
    """
    
    return np.sqrt(kb * T / (mu * mp))        
                
    
def T_freeze(mx, Ex, nx, Nx = 1e15, fx = 1):
    
    """
    Input
    -----
    mx:
        molecular weight in proton masses
    Ex:
        binding energy in K
    nx:
        number density in cm^-3
    Nx:
        number of adsorption sites per cm^2 (default 1e15)
    fx:
        fraction of occupied adsorption sites (default 1)
        
    Output
    ------
    Tfreeze:
        freezing temperature in K
    """
    
    def Tfreezefn(x):
        
        """
        Auxiliary function, defined as Tfreezefn(x) = f(x) - x,
        where "x" stands for temperature and f(x) is the expression
        on the RHS of equation (4) in Hollenbach et al. (2009).
        Needed to find the freezing temperature Tf from 
        Tfreezefn(Tf) = 0 (i.e., f(Tf) = Tf), because the expression 
        for T_freeze in equation (4) also depends on Tf through the
        thermal velocity. 
        """
        
        return Ex * (np.log(4 * Nx * fx * vib_freq(Ex, mx) / \
            (nx * v_thermal(x, mx))))**(-1) - x
            
    return brentq(Tfreezefn, 0.01, 1000) #solves for Tfreezefn(x) = 0
                                         #as explained above 
    
    
    
