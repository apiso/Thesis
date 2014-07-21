import numpy as np
from T_freeze import T_freeze

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


#in the following we calculate T_freeze for various species
T_freeze_H20 = T_freeze(18., 5800, 1e6)
T_freeze_CO2 = T_freeze(44., 2000, 3e5)
T_freeze_CO = T_freeze(18., 850, 1e6)


def n(T, elem):
    
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
        return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i] + n_C_O('H2O')[i], \
            n_C_O('C_grains')[i] + n_C_O('silicate')[i]])
            
    elif T_freeze_H20 >= T >= T_freeze_CO2:
        return np.array([n_C_O('CO')[i] + n_C_O('CO2')[i], \
            n_C_O('H2O')[i] + n_C_O('C_grains')[i] + n_C_O('silicate')[i]])
            
    elif T_freeze_CO2 > T > T_freeze_CO:
        return np.array([n_C_O('CO')[i], \
            n_C_O('C_grains')[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
                n_C_O('CO2')[i]])
                
    elif T_freeze_CO > T:
        return np.array([0, n_C_O('C_grains')[i] + n_C_O('silicate')[i] + n_C_O('H2O')[i] + \
                n_C_O('CO2')[i] + n_C_O('CO')[i]])
            
            

def C_O_ratio(a, T0 = 200, q = 0.62):
    
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
    
    T = T0 * a**(-q)
    
    return n(T, 'C') / n(T, 'O')
                       
            
            
            