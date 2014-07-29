from utils.constants import G, kb, mp, Msun, cmperau
import numpy as np
from T_freeze import T_freeze, Rdes, tevap, vib_freq
from C_to_O import T_freeze_H20, T_freeze_CO2, T_freeze_CO
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
    """Stopping time in seconds with r in AU and s in cm"""

    if eps == 0:
         if s <= 9 * lambdamfp(r, Sigma0, betaS, T0, betaT, mu) / 4:

            return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                 cdisk(r, T0, betaT, mu))
                 
         else:
              return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                 cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))
    elif eps == 1:
         return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                 cdisk(r, T0, betaT, mu))
    elif eps == -1:
         return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                 cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))
        
def taus(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
    """Dimensionless stopping time with r in AU and s in cm"""
    return ts(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) * Omegak(r, Mstar) 

def n(betaS = 3./2, betaT = 3./7):
    """Power law coefficient in P \propto r^(-n)"""
    return betaS + betaT / 2. + (3./2)    
            
def eta(r, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2):
     """Dimensionless correction coefficient eta = vk - vgas"""
     return n(betaS, betaT) * cdisk(r, T0, betaT, mu)**2 / \
        (2 * Omegak(r, Mstar) * r * cmperau)**2  
        
def tr(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
    """Radial drift time in seconds with r in AU and s in cm"""
    return (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
        Mstar, sigma, eps)**2)/ \
            taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
                Mstar, sigma, eps) / (2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar))

def rdot(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     return - (r * cmperau) / tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma)


################################################################################

def tdes(mx, Ex, Tx, s, Nx = 1e15, rhos = 3.0):

#     return rhos * (4 * np.pi * s**3 / 3) / (mx * mp) * \
#            1. / (Rdes(mx, Ex, Tx) * Nx * 4 * np.pi * s**2)

     return rhos / (3 * mx * mp) * s / (Nx * Rdes(mx, Ex, Tx)) 




def rf(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, tin = 0):

     #rdot = r * cmperau / tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma)

     def f(x, t):
          
          return np.array([ \
               rdot(x[0] / cmperau, x[1], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma), \
               - 3 * mx * mp  / rhos * \
                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])


          

     tv = np.linspace(tin, (3e6 * 365 * 24 * 3600), npts)
     y = odeint(f, [rin * cmperau, sin], tv)

##     for i in range(len(y[:,0]) - 1):
##          if y[:,0][i] <= y[:,0][i + 1]:
##               break
##          #sf = y[:,1][i - 1]
##          #npts = 10 * npts
     fint = interp1d(y[:,1][::-1], y[:,0][::-1])

     try: 
          return float(fint(0)) / cmperau
     except ValueError:
          return y[:,0][-1] / cmperau
     #Re = s * vrel / (lambdamfp(rH2O) * cdisk(rH2O))
     #return y[:,0] / cmperau, y[:,1]


def r_stop(s, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     def f(r):
          return tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma) - \
                 tdes(mx, Ex, Tdisk(r, T0, betaT), s, Nx, rhos)

     return brentq(f, 1e-3, 1e2)


def r_freeze(mx, Ex, nx, T0 = 120., betaT = 3./7):

     def f(x):
          return Tdisk(x, T0, betaT) - T_freeze(mx, Ex, nx)

     return brentq(f, 0.1, 100)









     




        
        
