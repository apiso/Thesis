from utils.constants import G, kb, mp, Msun, cmperau
import numpy as np
from T_freeze import T_freeze, Rdes, tevap, vib_freq
from C_to_O import T_freeze_H20, T_freeze_CO2, T_freeze_CO
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from utils.zbrac import zbrac

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
    
####def ts(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
####    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
####    """Stopping time in seconds with r in AU and s in cm"""
####
####    if eps == 0:
####         if s <= 9 * lambdamfp(r, Sigma0, betaS, T0, betaT, mu) / 4:
####
####            return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
####                 cdisk(r, T0, betaT, mu))
####                 
####         else:
####              return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
####                 cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))
####    elif eps == 1:
####         return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
####                 cdisk(r, T0, betaT, mu))
####    elif eps == -1:
####         return 4 * rhos * s**2 / (9 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
####                 cdisk(r, T0, betaT, mu) * lambdamfp(r, Sigma0, betaS, T0, betaT, mu, Mstar, sigma))


def n(betaS = 3./2, betaT = 3./7):
    """Power law coefficient in P \propto r^(-n)"""
    return 1 #betaS + betaT / 2. + (3./2)    
            
def eta(r, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2):
     """Dimensionless correction coefficient eta = vk - vgas"""
     return n(betaS, betaT) * cdisk(r, T0, betaT, mu)**2 / \
        (2 * (Omegak(r, Mstar) * r * cmperau)**2)  

def ts(r, s, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), eps = 0):
    """Stopping time in seconds with r in AU and s in cm"""

    if eps == 0:
         if s <= 9 * lambdamfp(r, Sigma0, betaS, T0, betaT, mu) / 4:

            return rhos * s / (rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                 vth(r, T0, betaT, mu))
                 
         else:

             def f(t):
                  
             
                  vroverr = -2 * eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
                            (t* Omegak(r, Mstar)) / (1 + t * Omegak(r, Mstar))
                  vphioverr = - eta(r, T0, betaT, mu, Mstar, betaS) * Omegak(r, Mstar) * \
                         (1. / (1 + (t * Omegak(r, Mstar))**2) - 1)
                  vrel = np.sqrt(vroverr**2 + vphioverr**2)

                  Re = 4 * vrel * s / \
                        (lambdamfp(r, Sigma0, betaS, T0, betaT, mu) * vth(r, T0, betaT, mu))

                  CD = 24.0/Re * (1.0+0.27*Re)**0.43 + 0.47 * (1.0 - np.exp(-0.04 * Re**0.38))
             

                  return 8 * rhos * s / (3 * rhodisk(r, Sigma0, betaS, T0, betaT, mu, Mstar) * \
                           vrel * CD) - t

             time = brentq(f, 1e-20, 1e30)

             return time
          
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





          
####
####def rf(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
####    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 0, eps = 0, vphi = 0):
####
####     def f(x, t):
####          
####          return np.array([ \
####               rdot(x[0] / cmperau, x[1], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi), \
####               - 3 * mx * mp  / rhos * \
####                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])
####          
####
####     tv = np.linspace(tin, (3e6 * 365 * 24 * 3600), npts)
####     y = odeint(f, [rin * cmperau, sin], tv)
####
####     if y[:,1][-1] > 0:
####          return tv, y[:,0] / cmperau, y[:,1]
####
####     else:
####          for i in range(len(y[:,0]) - 1):
####               if y[:,1][i] >=0 and y[:,1][i + 1] < 0:
####                    break
####          tint = np.linspace(tv[i], tv[i + 1], nptsin)
####	  yint = odeint(f, [y[:,0][i], y[:,1][i]], tint)
####
####	  finta = interp1d(yint[:,1][::-1], yint[:,0][::-1])
####	  fintt = interp1d(yint[:,1][::-1], tint[::-1])
####
####	  af = float(finta(0))
####	  tf = float(fintt(1))
####
####
####	  snew = np.linspace(sin, 0, nptsin)
####	  anew, tnew = [], []
####
####	  for i in range(len(snew)):
####               anew = np.append(anew, float(finta(snew[i])))
####               tnew = np.append(tnew, float(fintt(snew[i])))
####
####          return tnew, anew / cmperau, snew

	  
          




          

####def rf(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
####    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 0, eps = 0, vphi = 0):
####
####     def f(x, t):
####          
####          return np.array([ \
####               rdot(x[0] / cmperau, x[1], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi), \
####               - 3 * mx * mp  / rhos * \
####                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])
####
####     tv = np.linspace(tin, 3 * 10**6 * 365 * 24 * 3600, npts)
####
####     y = odeint(f, [rin * cmperau, sin], tv)
####
####     if y[:,1][-1] > 0:
####          return tv, y[:,0] / cmperau, y[:,1]
####
####     else:
####
####
####          def delta(t):
####               
####               #tv = np.linspace(tin, t, npts)
####               
####               y = odeint(f, [rin * cmperau, sin], [tin, t])
####
####               s1 = y[:,1][-1]
####
####               return s1
####
####          ##t1, t2 = zbrac(delta, tdes(mx, Ex, Tdisk(rin, T0, betaT), sin, Nx, rhos), tr(rin, sin, rhos, T0, betaT, mu, Sigma0, betaS, \
####          ##         Mstar, sigma, eps, vphi))[0]
####          #x = zbrac(delta, 1e2, 1e14)
####          #t1, t2 = x[0]
####          #success = x[1]
####
####          
####          tmatch = brentq(delta, tdes(mx, Ex, Tdisk(rin, T0, betaT), sin, Nx, rhos) / 10, 3 * 10**6 * 365 * 24 * 3600)
####
####          tv = np.linspace(tin, tmatch, npts)
####               
####          y = odeint(f, [rin * cmperau, sin], tv)
####
####          return tv, y[:,0] / cmperau, y[:,1]
####


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


def rf(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0):

     def f(x, t):
          
          return np.array([ \
               rdot(x[0] / cmperau, x[1], rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi), \
               - 3 * mx * mp  / rhos * \
                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])
          

     tv = np.logspace(np.log10(tin), np.log10(3e6 * 365 * 24 * 3600), npts)
     y = odeint(f, [rin * cmperau, sin], tv)


     for i in range(len(y[:,0]) - 1):
          if y[:,1][i] >=0 and y[:,1][i + 1] < 0:
               break
          #sf = y[:,1][i - 1]
          #npts = 10 * npts
     if i == len(y[:,0] - 1):
          return tv, y[:,0] / cmperau, y[:,1]
     
          #return y[:,0][-1] / cmperau
     else:
	  tint = np.logspace(np.log10(tv[i]), np.log10(tv[i + 1]), nptsin)
	  yint = odeint(f, [y[:,0][i], y[:,1][i]], tint)

     	  finta = interp1d(yint[:,1][::-1], yint[:,0][::-1])
     	  fintt = interp1d(yint[:,1][::-1], tint)

     	  try:
              af = float(finta(0))
              tf = float(fintt(0))

              afv = np.append(y[:,0][:i], af)
              sfv = np.append(y[:,1][:i], 0)
              tfv = np.append(tv[:i], tf)
              
              return tfv, afv / cmperau, sfv
     	      #return float(fint(0)) / cmperau
          except ValueError:
               return tv, y[:,0] / cmperau, y[:,1]
               #return y[:,0][-1] / cmperau


def Mdot_solids(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0, dusttogas = 0.01):

     Sigmap = dusttogas * Sigmadisk(rin, Sigma0, betaS)
     v = - rdot(rin, sin, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma, eps, vphi)

     return v * Sigmap * 2 * np.pi * (rin * cmperau) / (Msun / (365 * 24 * 3600))


###################################################################################################

def write_drag_file(folder, s, na, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7,  mu = 2.35, Sigma0 = 2200, betaS = 3./2, Mstar = Msun, \
                    sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0, amin = 0.1, amax = 1e2, fact = 1000):

     f = open('../dat/'+folder+'/'+'s_' + str(s) + '_na_' + str(na) + '.txt', 'wb')

     a = np.logspace(np.log10(amin), np.log10(amax), na)

     for i in range(na):
          temp = rf(a[i], s, mx, Ex, Nx, rhos, T0, betaT , mu, Sigma0, betaS, \
              Mstar, sigma, npts, nptsin, tin, eps, vphi)
          n = len(temp[0])
          f.write('%s ' % str(a[i]))
          f.write(' %s \n' % str(n / fact + 1))
          for j in range(int(n / fact)):
               f.write(' %s ' % str(temp[0][j * fact])) #time elapsed in seconds
               f.write(' %s ' % str(temp[1][j * fact])) #desorption distance in AU
               f.write(' %s \n' % str(temp[2][j * fact])) # size in cm
          f.write(' %s ' % str(temp[0][-1])) #time elapsed in seconds
          f.write(' %s ' % str(temp[1][-1])) #desorption distance in AU
          f.write(' %s \n' % str(temp[2][-1]))

     f.close()



def read_drag_file(filename, folder, na, nmax = 1e3):
     f = open('../dat/'+folder+'/'+filename, 'r')
     array = 0 * np.ndarray(shape = (na, nmax, 3), dtype = float)
     a = []
     for i in range(na):
          line = f.readline()
          a = np.append(a, float(line.split()[0]))
          npts = int(line.split()[1])
          for j in range(npts):
               line = f.readline()
               array[i, j, 0] = float(line.split()[0])
               array[i, j, 1] = float(line.split()[1])
               array[i, j, 2] = float(line.split()[2])

     #for i in range(na):
     #     filter(lambda a: a != 0.0, array[i, :, 0])

     return a, array


#def remove_zeros(filename, folder, na, nmax = 1e3):

#     a, arr = read_drag_file(filename, folder, na, nmax = nmax)

##
##     for line in f:
##    #line = f.readline()
##         a = np.append(a, float(line.split()[0]))
##         s = np.append(s, float(line.split()[1]))
##     f.close()

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
     





     




        
        
