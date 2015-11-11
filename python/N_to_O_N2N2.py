"""This script is used to calculate the C/O ratio in gas and in dust
as a function of temperature, i.e. the location in the disk relative 
to the H2O, CO2 and CO snowlines"""

import numpy as np
from T_freeze import T_freeze
from drift_timescales_active_Tdisk import Tdisk
from utils.constants import Msun

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


#in the following we calculate T_freeze for various species
T_freeze_H20 = T_freeze(18., 5800, 1e6)
T_freeze_NH3 = T_freeze(17., 2965., 5.5e4)
T_freeze_CO2 = T_freeze(44., 2000, 3e5)
T_freeze_CO = T_freeze(28., 834., 1e6)
T_freeze_N2 = T_freeze(28., 767., 0.8e6/2)
#T_freeze_NH3 = T_freeze(17., 2965., 1.5371e5)



def n(T, elem, NH3mid = 0, NH3max = 0):
    
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
        
    if T >= T_freeze_H20:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('H2O')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i], \
            n_N_O('C_grains')[i] + n_N_O('silicate')[i]])

    elif T_freeze_H20 >= T >= T_freeze_NH3:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i], \
            n_N_O('H2O')[i] + n_N_O('C_grains')[i] + n_N_O('silicate')[i]])            
                                    
    elif T_freeze_NH3 >= T >= T_freeze_CO2:
        return np.array([n_N_O('CO')[i] + n_N_O('CO2')[i] + n_N_O('N2', NH3mid, NH3max)[i], \
            n_N_O('H2O')[i] + n_N_O('NH3', NH3mid, NH3max)[i] + n_N_O('C_grains')[i] + n_N_O('silicate')[i]])
            
    elif T_freeze_CO2 >= T >= T_freeze_CO:
        return np.array([n_N_O('CO')[i] + n_N_O('N2', NH3mid, NH3max)[i], \
            n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('NH3', NH3mid, NH3max)[i]])
                
    elif T_freeze_CO >= T >= T_freeze_N2:
        return np.array([n_N_O('N2', NH3mid, NH3max)[i], n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('CO')[i] + n_N_O('NH3', NH3mid, NH3max)[i]])

    elif T_freeze_N2 > T:
        return np.array([0, n_N_O('C_grains')[i] + n_N_O('silicate')[i] + n_N_O('H2O')[i] + \
                n_N_O('CO2')[i] + n_N_O('CO')[i] + n_N_O('N2', NH3mid, NH3max)[i] + n_N_O('NH3', NH3mid, NH3max)[i]])
            
            

def N_O_ratio(a, alpha, Mdot, k0, mu, Mstar = Msun, T0 = 200, q = 0.62, acc = 0, NH3mid = 0, NH3max = 0):
    
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
    
    return n(T, 'N', NH3mid, NH3max) / n(T, 'O', NH3mid, NH3max)
                       
            
            
            
