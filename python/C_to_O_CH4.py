"""This script is used to calculate the C/O ratio in gas and in dust
as a function of temperature, i.e. the location in the disk relative 
to the H2O, CO2 and CO snowlines"""

import numpy as np
from T_freeze import T_freeze
from drift_timescales_active_Tdisk import Tdisk
from utils.constants import Msun

def n_C_O(species, CH4mid = 0, CH4max = 0):
    
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
    elif species == 'C_grains' and CH4mid == 0 and CH4max == 0:
        return 0.6, 0
    elif species == 'C_grains' and CH4mid == 1 and CH4max == 0:
        return 0.6 - 0.0555 * 0.9, 0
    elif species == 'C_grains' and CH4mid == 0 and CH4max == 1:
        return 0.6 - 0.13 * 0.9, 0
    elif species == 'silicate':
        return 0, 1.4
    elif species == 'CH4' and CH4mid == 0 and CH4max == 0:
        return 0, 0
    elif species == 'CH4' and CH4mid == 1 and CH4max == 0:
        return 0.0555 * 0.9, 0 #5.55% H2O (Oberg+08 Spitzer paper) 
    elif species == 'CH4' and CH4mid == 0 and CH4max == 1:
        return 0.13 * 0.9, 0 #13% H2O (Oberg+08 c2d paper)
     


#in the following we calculate T_freeze for various species
T_freeze_H20 = T_freeze(18., 5800, 1e6)
T_freeze_CO2 = T_freeze(44., 2000, 3e5)
T_freeze_CH4 = T_freeze(16., 1300., 0.0555*1e6) #binding energy from Garrod+06 A&A paper
T_freeze_CO = T_freeze(28., 1388, 1e6)
#T_freeze_CH4 = T_freeze(16., 1300., 0.13*1e6)


def n(T, elem, CH4mid, CH4max):
    
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
        
    if T >= T_freeze_H20:
        return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i] + n_C_O('H2O')[i] + n_C_O('CH4', CH4mid, CH4max)[i], \
            n_C_O('C_grains', CH4mid, CH4max)[i] + n_C_O('silicate')[i]])
            
    elif T_freeze_H20 >= T >= T_freeze_CO2:
        return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i] + n_C_O('CH4', CH4mid, CH4max)[i], \
            n_C_O('H2O')[i] + n_C_O('C_grains', CH4mid, CH4max)[i] + n_C_O('silicate')[i]])

    elif T_freeze_CO2 >= T > T_freeze_CO:
        return np.array([n_C_O('CO')[i] + n_C_O('CH4', CH4mid, CH4max)[i], \
            n_C_O('C_grains', CH4mid, CH4max)[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
                n_C_O('CO2')[i]])            
            
    elif T_freeze_CO >= T > T_freeze_CH4:
        return np.array([n_C_O('CH4', CH4mid, CH4max)[i], \
            n_C_O('C_grains', CH4mid, CH4max)[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
                n_C_O('CO2')[i] + +n_C_O('CO')[i]])
                
    elif T_freeze_CH4 >= T:
        return np.array([0, n_C_O('C_grains', CH4mid, CH4max)[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
                n_C_O('CO2')[i] + n_C_O('CO')[i] + n_C_O('CH4', CH4mid, CH4max)[i]])
            
            

def C_O_ratio(a, alpha, Mdot, k0, mu, Mstar = Msun, T0 = 200, q = 0.62, acc = 0, CH4mid = 0, CH4max = 0):
    
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
    if acc == 0:
        T = T0 * a**(-q)
    else:
        T = Tdisk(a, alpha, Mdot, k0, mu, Mstar, T0, q)
    
    return n(T, 'C', CH4mid, CH4max) / n(T, 'O', CH4mid, CH4max)
                       
            
            
            
