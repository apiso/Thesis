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

def vth(r, T0 = 120, betaT = 3./7, mu = 2.35):
    """Mean thermal velocity for a Maxwellian distribution in cm s^-1 with r in AU"""
    return np.sqrt(8 / np.pi) * cdisk(r, T0, betaT, mu)
    
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

##def vrel(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
##    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
##
##     vroverr = -2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
##                  taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) / \
##                  (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2)
##     vphioverr = - eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
##                    (1. / (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2) - 1)
##     return np.sqrt(vroverr**2 + vphioverr**2)
    
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

##def ts(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
##    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
##    """Stopping time in seconds with r in AU and s in cm"""
##
##    if eps == 0:
##         if s <= 9 * lambdamfp(r, Sigma0, betaS, T0, betaT, mu) / 4:
##
##            return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
##                 cdisk(r, T0, betaT, mu))
##                 
##         else:
##             vroverr = -2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
##                  taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) / \
##                  (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2)
##             vphioverr = - eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
##                    (1. / (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2) - 1)
##             vrel = np.sqrt(vroverr**2 + vphioverr**2)
##
##             Re = 4 * vrel * s / \
##                   (lambdamfp(r, Sigma0, betaS, T0, betaT, mu) * vth(r, T0, betaT, mu))
##
##             CD = 24.0/Re * (1.0+0.27*Re)**0.43 + 0.47 * (1.0 - np.exp(-0.04 * Re**0.38))
##              
##             return 8 * rhos * s / (3 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
##                 vrel(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) * CD)
##    elif eps == 1:
##         return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
##                 cdisk(r, T0, betaT, mu))
##    elif eps == -1:
##         return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
##                 cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))
        
def taus(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
    """Dimensionless stopping time with r in AU and s in cm"""
    return ts(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) * Omegak(r, Mstar) 

def n(betaS = 3./2, betaT = 3./7):
    """Power law coefficient in P \propto r^(-n)"""
    return 1 #betaS + betaT / 2. + (3./2)    
            
def eta(r, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2):
     """Dimensionless correction coefficient eta = vk - vgas"""
     return n(betaS, betaT) * cdisk(r, T0, betaT, mu)**2 / \
        (2 * (Omegak(r, Mstar) * r * cmperau)**2)  
        
def tr(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0, vphi = 0):
    """Radial drift time in seconds with r in AU and s in cm"""
    if vphi == 0:
        return (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
            Mstar, sigma, eps)**2)/ \
                taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, \
                    Mstar, sigma, eps) / (2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar))
    else:
        vroverr = -2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
                  taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) / \
                  (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2)
        vphioverr = - eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
                    (1. / (1 + taus(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2) - 1)
        return 1. / np.sqrt(vroverr**2 + vphioverr**2)

def rdot(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0, vphi = 0):

     return - (r * cmperau) / tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi)

def t_gas_acc(r, alpha, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2):
     return 1. / (2 * alpha * Omegak(r, Mstar) * eta(r, T0, betaT, mu, Mstar, betaS))


################################################################################

def tdes(mx, Ex, Tx, s, Nx = 1e15, rhos = 3.0):

#     return rhos * (4 * np.pi * s**3 / 3) / (mx * mp) * \
#            1. / (Rdes(mx, Ex, Tx) * Nx * 4 * np.pi * s**2)

     return rhos / (3 * mx * mp) * s / (Nx * Rdes(mx, Ex, Tx)) 




def rf(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 0, eps = 0, vphi = 0):

     #rdot = r * cmperau / tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma)

     def f(x, t):
          
          return np.array([ \
               rdot(x[0] / cmperau, x[1], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi), \
               - 3 * mx * mp  / rhos * \
                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])


          

     tv = np.linspace(tin, (3e6 * 365 * 24 * 3600), npts)
     y = odeint(f, [rin * cmperau, sin], tv)

##     vroverr, vphioverr, Re = [], [], []
##
##     for i in range(len(y[:,0])):
##     
##          vroverr = np.append(vroverr, -2 * eta(y[:,0][i], T0, betaT, mu, Mstar, betaS) * Omegak(y[:,0][i], Mstar) * \
##               taus(y[:,0][i], y[:,1][i], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps) / \
##                  (1 + taus(y[:,0][i], y[:,1][i], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2))
##          vphioverr = np.append(vphioverr, - eta(y[:,0][i], T0, betaT, mu, Mstar, betaS) * Omegak(y[:,0][i], Mstar) * \
##               (1. / (1 + taus(y[:,0][i], y[:,1][i], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps)**2) - 1))
##     vrel = np.sqrt(vroverr**2 + vphioverr**2)
##
##     for i in range(len(y[:,0])):
##     
##          Re = np.append(Re, y[:,1][i] * vrel[i] / (cdisk(y[:,0][i] / cmperau, T0, betaT, mu) * \
##                           lambdamfp(y[:,0][i] / cmperau, Sigma0, betaS, T0, betaT, mu, Mstar, sigma)))

     for i in range(len(y[:,0]) - 1):
          if y[:,1][i] >=0 and y[:,1][i + 1] < 0:
               break
          #sf = y[:,1][i - 1]
          #npts = 10 * npts
     if i == len(y[:,0] - 1):
          return y[:,0][-1] / cmperau
     else:
	  tint = np.linspace(tv[i], tv[i + 1], nptsin)
	  y = odeint(f, [y[:,0][i], y[:,1][i]], tint)

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

     try:
          return brentq(f, 1e-3, 1e2)
     except ValueError:
          return 1e-10


def r_freeze(mx, Ex, nx, T0 = 120., betaT = 3./7):

     def f(x):
          return Tdisk(x, T0, betaT) - T_freeze(mx, Ex, nx)

     return brentq(f, 0.1, 100)


###################################################################################################

def read_drag_file(filename, folder):

     f = open('../dat/'+folder+'/'+filename, 'r')
     a, s = [], []

     for line in f:
    #line = f.readline()
         a = np.append(a, float(line.split()[0]))
         s = np.append(s, float(line.split()[1]))
     f.close()

     return a, s

def read_drag_file_many(filename, folder, ns, na):
     f = open('../dat/'+folder+'/'+filename, 'r')

     array = 0 * np.ndarray(shape = (ns, na), dtype = float)
     s = []
     a = []
     
     for i in range(ns):
          line = f.readline()
          s = np.append(s, float(line.split()[0]))
          for j in range(na):
               line = f.readline()
               a, array[i, j] = np.append(a, float(line.split()[0])), float(line.split()[1])
     a = a[:na]
               
     return s, a, array
     





     




        
        
