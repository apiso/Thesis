"""This script is used to calculate the C/O ratio in gas and in dust
as a function of temperature, i.e. the location in the disk relative 
to the H2O, CO2 and CO snowlines"""

import numpy as np
from T_freeze import T_freeze
from drift_timescales_active_Tdisk import Tdisk
from utils.constants import Msun
from drift_timescales import read_drag_file_many

def n_N_O(species, NH3mid = 0, NH3max = 0):
    
    """
        
    Returns the abundance of N and O in various molecular species
    (numbers are taken from Oberg+11b, Table 1)
    
    
    Input
    -----
    species:
        string: 'CO', 'CO2' etc.
    
    Output
    ------
    n_N:
        abundance of N with respect to H2
    n_O:
        abundance of O with respect to H2
        
    """
    
    if species == 'CO':
        return 0, 1.5
    elif species == 'CO2':
        return 0, 0.6
    elif species == 'H2O':
        return 0, 0.9
    elif species == 'C_grains':
        return 0, 0
    elif species == 'silicate':
        return 0, 1.4
    elif species == 'N2' and NH3mid == 0 and NH3max == 0:
        return 0.8, 0
    elif species == 'N2' and NH3mid == 1 and NH3max == 0:
        return (0.8 - 0.9 * 0.055), 0
    elif species == 'N2' and NH3max == 1 and NH3mid == 0:
        return (0.8 - 0.9 * 0.1537), 0
    elif species == 'NH3' and NH3mid == 0 and NH3max == 0:
        return 0, 0
    elif species == 'NH3' and NH3mid == 1 and NH3max == 0:
        return 0.055 * 0.9, 0 #0.055 * n_H2O from Oberg+11 Spitzer paper
    elif species == 'NH3' and NH3max == 1 and NH3mid == 0:
        return 0.1537 * 0.9, 0 #0.1537 * n_H2O from Bottinelli+10 Spitzer paper


s, a, arrayH2O = read_drag_file_many('s_neg3_8_steady_correct.txt', 'H2O', 50, 50)
s, a, arrayCO2 = read_drag_file_many('s_neg3_8_steady_correct.txt', 'CO2', 50, 50)
s, a, arrayCO = read_drag_file_many('s_neg3_8_steady_correct_CO_CO.txt', 'CO', 50, 50)
s, a, arrayN2 = read_drag_file_many('s_neg3_8_steady_correct_N2_N2.txt', 'N2', 50, 50)
s, a, arrayNH3 = read_drag_file_many('s_neg3_8_steady_correct.txt', 'NH3', 50, 50)


def n(index, r, elem, NH3mid = 0, NH3max = 0):
    
    """
    
    Returns the gaseous and solid abundance of a given element (N, O)
    as a function of temperature, e.g. between different snowlines
    
    Input
    -----
    T:
        disk midplane temperature in units of K
    elem:
        string: 'N' or 'O'
        
    Output
    ------
    n_gas:
        abundance of element 'elem' in gaseous form relative to H2 abundance
    n_solid:
        abundance of element 'elem' in solid form relative to H2 abundance
    
    """
    
    if elem == 'N':
        i = 0 #flag to simplify the use of the n_N_O function; = 0 for N
    elif elem == 'O':
        i = 1 #flag to simplify the use of the n_N_O function; = 1 for O
        
    size = s[index]
    aH2O = arrayH2O[index][-1]
    aCO2 = arrayCO2[index][-1]
    aCO = arrayCO[index][-1]
    aN2 = arrayN2[index][-1]
    aNH3 = arrayNH3[index][-1]
        
    if r <= aH2O:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('H2O')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i], \
            n_N_O('C_grains')[i] + n_N_O('silicate')[i]]), size

    elif aH2O <= r <= aNH3:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i], \
            n_N_O('H2O')[i] + n_N_O('C_grains')[i] + n_N_O('silicate')[i]]), size            
                                    
    elif aNH3 <= r <= aCO2:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('N2', NH3mid, NH3max)[i], \
            n_N_O('H2O')[i] + n_N_O('NH3', NH3mid, NH3max)[i] + n_N_O('C_grains')[i] + n_N_O('silicate')[i]]), size
            
    elif aCO2 <= r <= aCO:
        return np.array([n_N_O('CO')[i] + n_N_O('N2', NH3mid, NH3max)[i], \
            n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('NH3', NH3mid, NH3max)[i]]), size
                
    elif aCO <= r <= aN2:
        return np.array([n_N_O('N2', NH3mid, NH3max)[i], n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('CO')[i] + n_N_O('NH3', NH3mid, NH3max)[i]]), size

    elif aN2 < r:
        return np.array([0, n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('CO')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i]]), size
            
            

def N_O_ratio(r, index, NH3mid = 0, NH3max = 0):
    
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
    
    return n(index, r, 'N', NH3mid, NH3max)[0] / n(index, r, 'O', NH3mid, NH3max)[0], n(index, r, 'O', NH3mid, NH3max)[1]
            
