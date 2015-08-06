"""This script is used to calculate the C/O ratio in gas and in dust
as a function of temperature, i.e. the location in the disk relative 
to the H2O, CO2 and CO snowlines"""

import numpy as np
from T_freeze import T_freeze
from drift_timescales import read_drag_file_many
from drift_timescales_active_Tdisk import Mdots, Sigmadisk
from utils.constants import Msun

def n_C_O(species):
    
    """
        
    Returns the abundance of C and O in various molecular species
    (numbers are taken from Oberg+11b, Table 1)
    
    
    Input
    -----
    species:
        string: 'CO', 'CO2' etc.
    
    Output
    ------
    n_C:
        abundance of C with respect to H2
    n_O:
        abundance of O with respect to H2
        
    """
    
    if species == 'CO':
        return 1.5, 1.5
    elif species == 'CO2':
        return 0.3, 0.6
    elif species == 'H2O':
        return 0, 0.9
    elif species == 'C_grains':
        return 0.6, 0
    elif species == 'silicate':
        return 0, 1.4
mC = 12
mO = 16
mH = 1
mSi = 28

f = 0.01

Mtot = (0.9*mC + 0.9*mO + 0.3*mC + 0.6*mO + 0.45*mH + 0.9*mO + 0.6*mC + 1.4*mO + 0.7*mSi)
fCO = (0.9*mC + 0.9*mO) * f / Mtot
fCO2 = (0.3*mC + 0.6*mO) * f / Mtot
fH2O = (0.9*mO + 0.45*mH) * f / Mtot
fCgr = (0.6*mC) * f / Mtot
fSiO2 = (1.4*mO + 0.7*mSi) * f / Mtot

#in the following we calculate T_freeze for various species
s, a, arrayH2O = read_drag_file_many('s_neg3_8_steady_correct.txt', 'H2O', 50, 50)
s, a, arrayCO2 = read_drag_file_many('s_neg3_8_steady_correct.txt', 'CO2', 50, 50)
s, a, arrayCO = read_drag_file_many('s_neg3_8_steady_correct.txt', 'CO', 50, 50)




def n(index, r, elem, alpha, Mdot, k0, rhos = 3.0, dr = 1e-3, Mstar = Msun, mu = 2.35, T0 = 120, betaT = 3./7, f = 0.01):
    
    """
    
    Returns the gaseous and solid abundance of a given element (C, O)
    as a function of temperature, e.g. between different snowlines
    
    Input
    -----
    T:
        disk midplane temperature in units of K
    elem:
        string: 'C' or 'O'
        
    Output
    ------
    n_gas:
        abundance of element 'elem' in gaseous form relative to H2 abundance
    n_solid:
        abundance of element 'elem' in solid form relative to H2 abundance
    
    """
    
    if elem == 'C':
        i = 0 #flag to simplify the use of the n_C_O function; = 0 for C
    elif elem == 'O':
        i = 1 #flag to simplify the use of the n_C_O function; = 1 for O
    
    size = s[index]
    aH2O = arrayH2O[index][-1]
    aCO2 = arrayCO2[index][-1]
    aCO = arrayCO[index][-1]  
    
    f2 = Mdots(aCO, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT) / Mdot
    f2 = Mdots(aCO, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT) / Mdot
    f2 = Mdots(aCO, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT) / Mdot
    
    MdotCO_CO = (fCO/f) * Mdots(aCO, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT)
    fCO_2 = MdotCO_CO / Mdot
    fCO_3 = fCO_2 
    
    MdotCO2_CO = (fCO2/f) * Mdots(aCO, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT)
    fCO2_2 = MdotCO2_CO / Mdot
    
    MdotCO2_CO2 = (fCO2_2/f2) * Mdots(aCO2, size, alpha, Mdot, k0, rhos, dr, Mstar, mu, T0, betaT)
    fCO_2 = MdotCO_CO / Mdot
    fCO_3 = fCO_2      
                
    if r > aCO:
                  
        SigmaCO = np.append(0, fCO * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT))
        SigmaCO2 = np.append(0, fCO2 * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT))
        SigmaH2O = np.append(0, fH2O * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT))
        SigmaCgr = np.append(0, fCgr * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT))
        SigmaSiO2 = np.append(0, fSiO2 * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT))
        
        return SigmaCO, SigmaCO2, SigmaH2O, SigmaCgr, SigmaSiO2
        
    elif aCO2 <= r <= aCO: #Zone 2 (between CO2 and CO snowlines)      

        SigmaCO = np.append((MdotCO_CO/Mdot) * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT), 0) 
        return SigmaCO 
        
    elif aH2O < r < aCO2: #Zone 3 (between H2O and CO2 snowlines)
        SigmaCO = np.append(fCO_2 * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT), 0)
    
        return SigmaCO 
        
    elif r <= aH2O:
        SigmaCO = np.append(fCO_3 * Sigmadisk(r, alpha, Mdot, k0, Mstar, mu, T0, betaT), 0)
         
        return SigmaCO         
            
    #elif aH2O <= r <= aCO2:
    #    return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i], \
    #        n_C_O('H2O')[i] + n_C_O('C_grains')[i] + n_C_O('silicate')[i]]), size

#elif r <= aH2O:
#        return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i] + n_C_O('H2O')[i], \
#            n_C_O('C_grains')[i] + n_C_O('silicate')[i]]), size
            
    #elif aCO2 < r < aCO:
    #    return np.array([n_C_O('CO')[i], \
    #        n_C_O('C_grains')[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
    #            n_C_O('CO2')[i]]), size
                
    #elif aCO < r:
    #    return np.array([0, n_C_O('C_grains')[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
    #            n_C_O('CO2')[i] + n_C_O('CO')[i]]), size
            
            

def C_O_ratio(r, index):
    
    """
    
    Returns the C/O ratio as a function of semimajor axis for a given disk
    temperature profile
    
    Input
    -----
    a:
        semimajor axis in AU
    T0:
        temperature normalization of disk profile in K; default 200
    q:
        power-law coefficient in disk temperature profile; default 0.62
        
    (note: T0 and q taken from Oberg+11b; temperature profile need not be a 
     power-law, or q may vary with a)
     
    Output
    ------
    C/O ratio in gas
    C/O ratio in grains
    
    """
    
    return n(index, r, 'C')[0] / n(index, r, 'O')[0], n(index, r, 'O')[1]
                       
            
            
            
