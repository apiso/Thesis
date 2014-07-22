from utils.constants import G, kb, mp, Msun, cmperau
import numpy as np

def Tdisk(r, T0 = 120, betaT = 3./7):
     """Disk temperature in K with r in AU"""
     return T0 * r**(-betaT) 
     
def Sigmadisk(r, Sigma0 = 2200, betaS = 3./2):
    """Disk gas surface density in g cm^-2 with r in AU"""
    return Sigma0 * r**(-betaS)
    
def cdisk(r, T0 = 120, betaT = 3./7, mu = 2.35):
    """Sound speed in cm s^-1 with r in AU"""
    return np.sqrt(kb * Tdisk(r, T0, betaT) / (mu * mp))
    
def Omegak(r, Mstar = Msun):
    """Keplerian angular frequency in s^-1 with r in AU"""
    return np.sqrt(G * Mstar / (r * cmperau)**3)

def Hdisk(r, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun):
    """Disk scale height in cm with r in AU"""
    return cdisk(r, T0, betaT, mu) / Omegak(r, Mstar)
    
def rhodisk(r, Sigma0 = 2200, betaS = 3./2, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun):
    """Disk gas density in g cm^-3 with r in AU"""
    return Sigmadisk(r, Sigma0, betaS) / \
        (np.sqrt(2 * np.pi) * Hdisk(r, T0, betaT, mu, Mstar))
        
def lambdamfp(r, Sigma0 = 2200, betaS = 3./2, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Mean free path in cm with r in AU"""
    return 1 / (np.sqrt(2) * sigma * \
        (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) / (mu * mp)))
    
def ts(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Stopping time in seconds with r in AU and s in cm"""
    if s <= 9 * lambdamfp(r, Sigma0, betaS, T0, betaT, mu) / 4:
       
        return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
            cdisk(r, T0, betaT, mu))
            
    else:
        return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
            cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))
        
def taus(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Dimensionless stopping time with r in AU and s in cm"""
    return ts(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma) * Omegak(r, Mstar) 

def n(betaS = 3./2, betaT = 3./7):
    """Power law coefficient in P \propto r^(-n)"""
    return betaS + betaT / 2. + (3./2)    
            
def eta(r, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2):
     """Dimensionless correction coefficient eta = vk - vgas"""
     return n(betaS, betaT) * cdisk(r, T0, betaT, mu)**2 / \
        (2 * Omegak(r, Mstar) * r * cmperau)**2  
        
def tr(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Radial drift time in seconds with r in AU and s in cm"""
    return (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
        Mstar, sigma)**2/ \
            taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
                Mstar, sigma)) * np.sqrt(G * Mstar * r * cmperau) / \
                    cdisk(r, T0, betaT, mu)**2
        
        
