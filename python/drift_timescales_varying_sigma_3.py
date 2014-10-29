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
     
def gamma(betaT = 3./7):
     
     return -betaT + 3./2

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


def nu(r, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun):

     return alpha * cdisk(r, T0, betaT, mu) * Hdisk(r, T0, betaT, mu, Mstar)

def Sigmadisk_act(r, t, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, r1 = 100 * cmperau, C = 4.45e18):

     gammad = gamma(betaT)
     nu1 = nu(r1 / cmperau, alpha, T0, betaT, mu, Mstar)
     ts = 1. / (3 * (2 - gamma(betaT))**2) * r1**2 / nu1
     T = t / ts + 1
     rtild = r * cmperau / r1

     return C / (3 * np.pi * nu1 * rtild**gammad) * T**(- (2.5 - gammad) / (2 - gammad)) \
            * np.exp(- rtild**(2 - gammad) / T)

def vacc_act(r, t, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, r1 = 100 * cmperau, C = 4.45e18):
    
    gammad = gamma(betaT)
    nu1 = nu(r1 / cmperau, alpha, T0, betaT, mu, Mstar)
    ts = 1. / (3 * (2 - gamma(betaT))**2) * r1**2 / nu1
    T = t / ts + 1
    rtild = r * cmperau / r1

    return 3 * alpha * kb * r**(-betaT) * rtild**(-1 - gammad) \
            * (-cmperau**2 * (-2 + gammad) * r**2 + (-2 + betaT + gammad) * rtild**gammad * r1**2 * T) * T0 / \
                    (mp * mu * Omegak(r, Mstar) * r1**3 * T)
    

    
def rhodisk(r, t, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, r1 = 100 * cmperau, C = 4.45e18):
    """Disk gas density in g cm^-3 with r in AU"""
    return Sigmadisk_act(r, t, alpha, T0, betaT, mu, Mstar, r1) / \
             (np.sqrt(2 * np.pi) * Hdisk(r, T0, betaT, mu, Mstar))

        
def lambdamfp(r, t, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, r1 = 100 * cmperau, C = 4.45e18, sigma = 2e-15):
    """Mean free path in cm with r in AU"""

    return 1 / (np.sqrt(2) * sigma * \
          (rhodisk(r, t, alpha, T0, betaT, mu, Mstar, r1, C) / (mu * mp)))




def n(betaS = 3./2, betaT = 3./7, Sigmad = 0):
    """Power law coefficient in P \propto r^(-n)"""
    return 1 #betaS + betaT / 2. + (3./2)    
            
def eta(r, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun):
     """Dimensionless correction coefficient eta = vk - vgas"""
     return cdisk(r, T0, betaT, mu)**2 / \
        (2 * (Omegak(r, Mstar) * r * cmperau)**2)  

def ts(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Stopping time in seconds with r in AU and s in cm"""


    if s <= 9 * lambdamfp(r, t, alpha, T0, betaT, mu, Mstar, r1, C, sigma) / 4:

       return rhos * s / (rhodisk(r, t, alpha, T0, betaT, mu, Mstar, r1, C) * \
            vth(r, T0, betaT, mu))
            
    else:

        def f(t):
             
        
             vroverr = -2 * eta(r, T0, betaT, mu, Mstar) * Omegak(r, Mstar) * \
                       (t* Omegak(r, Mstar)) / (1 + t * Omegak(r, Mstar))
             vphioverr = - eta(r, T0, betaT, mu, Mstar) * Omegak(r, Mstar) * \
                    (1. / (1 + (t * Omegak(r, Mstar))**2) - 1)
             vrel = np.sqrt(vroverr**2 + vphioverr**2)

             Re = 4 * vrel * s / \
                   (lambdamfp(r, t, alpha, T0, betaT, mu, Mstar, r1, C, sigma) * vth(r, T0, betaT, mu))

             CD = 24.0/Re * (1.0+0.27*Re)**0.43 + 0.47 * (1.0 - np.exp(-0.04 * Re**0.38))
        

             return 8 * rhos * s / (3 * rhodisk(r, t, alpha, T0, betaT, mu, Mstar, r1, C) * \
                      vrel * CD) - t

        time = brentq(f, 1e-20, 1e30)

        return time
          
        
def taus(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Dimensionless stopping time with r in AU and s in cm"""
    return ts(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) * Omegak(r, Mstar) 

        
def tr(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    """Radial drift time in seconds with r in AU and s in cm"""
    return (1 + taus(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma)**2)/ \
           taus(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) / \
           (2 * eta(r, T0, betaT, mu, Mstar) * Omegak(r, Mstar))


def rdot(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     return - (r * cmperau) / tr(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma)

def t_gas_acc(r, alpha, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2, Sigmad = 0):
     return 1. / (2 * alpha * Omegak(r, Mstar) * eta(r, T0, betaT, mu, Mstar, betaS, Sigmad))

#def rdot_gas(r, s, Mdotgas, Sigma0 = 2200, betaS = 3./2):
#
#     return - Mdotgas * Msun / (365 * 24 * 3600) / (Sigmadisk(r, Sigma0, betaS) * 2 * np.pi * (r * cmperau))

def rdot_with_acc(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     return rdot(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) + \
        vacc_act(r, t, alpha, T0, betaT, mu, Mstar, r1, C) / \
              (1 + taus(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma)**2)

#######################################################################################



def tdes(mx, Ex, Tx, s, Nx = 1e15, rhos = 3.0):

#     return rhos * (4 * np.pi * s**3 / 3) / (mx * mp) * \
#            1. / (Rdes(mx, Ex, Tx) * Nx * 4 * np.pi * s**2)
    return rhos / (3 * mx * mp) * s / (Nx * Rdes(mx, Ex, Tx)) 


def Mdot_solids(r, t, s, alpha, Sigmap, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     v = - rdot(r, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma)

     return v * Sigmap * 2 * np.pi * (r * cmperau) #/ (Msun / (365 * 24 * 3600))

def Mdot_gas(r, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

     v = - vacc_act(r, t, alpha, T0, betaT, mu, Mstar, r1, C)

     return v * Sigmadisk_act(r, t, alpha, T0, betaT, mu, Mstar, r1, C) * 2 * np.pi * (r * cmperau)


def dMgas_dt(rin, rout, t, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):

    return Mdot_gas(rout, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) - \
            Mdot_gas(rin, t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) 
          

def dMsol_dt(rin, rout, t, s, alpha, Sigmap_in, Sigmap_out, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15)):
    
    return Mdot_solids(rout, t, s, alpha, Sigmap_out, rhos, T0, betaT, mu, r1, C, Mstar, sigma) - \
            Mdot_solids(rin, t, s, alpha, Sigmap_in, rhos, T0, betaT, mu, r1, C, Mstar, sigma)



def dMdot(rin, rout, s, alpha, t0, tmax, n, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, f = 1e-4, constdtg = 0):
     
     Sigmap_in = Sigmadisk_act(rin, t0, alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas
     Sigmap_out = Sigmadisk_act(rout, t0, alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas

     t = np.linspace(t0, tmax, n)
     #t = np.logspace(np.log10(t0), np.log10(tmax), n)
     dM_gas, dM_sol, Sigmadv_in, Sigmadv_out, Sigmapv_in, Sigmapv_out = [], [], [], [], [Sigmap_in], [Sigmap_out]

     for i in range(len(t) - 1):

          dMgas = dMgas_dt(rin, rout, t[i], s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) * (t[i + 1] - t[i])
          dMsol = dMsol_dt(rin, rout, t[i], s, alpha, Sigmap_in, Sigmap_out, rhos, T0, betaT, mu, r1, C, Mstar, sigma) * (t[i + 1] - t[i])

          dM_gas = np.append(dM_gas, dMgas)
          dM_sol = np.append(dM_sol, dMsol)

          if constdtg == 0:
              Sigmap_in = Sigmap_in + dMsol / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
              Sigmap_out = Sigmap_out + dMsol / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
              
          else:
              Sigmap_in = Sigmadisk_act(rin, t[i], alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas
              Sigmap_out = Sigmadisk_act(rout, t[i], alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas

          Sigmadv_in = np.append(Sigmadv_in, Sigmadisk_act(rin, t[i], alpha, T0, betaT, mu, Mstar, r1, C))
          Sigmadv_out = np.append(Sigmadv_out, Sigmadisk_act(rout, t[i], alpha, T0, betaT, mu, Mstar, r1, C))
          
          Sigmapv_in = np.append(Sigmapv_in, Sigmap_in)
          Sigmapv_out = np.append(Sigmapv_out, Sigmap_out)

     return dM_gas * f, dM_sol, Sigmadv_in, Sigmadv_out, Sigmapv_in, Sigmapv_out



def Msol_dt(r1, r2, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0, dusttogas = 0.01, \
             Sigmad = 0, Sigmap = 0, Sigmap1 = 0, Sigmap2 = 0):
     if Sigmap == 0:
         return np.pi * (r2 * cmperau)**2 * dusttogas * Sigmadisk(r2, Sigma0, betaS) - \
                np.pi * (r1 * cmperau)**2 * dusttogas * Sigmadisk(r1, Sigma0, betaS)
     else:
         return np.pi * (r2 * cmperau)**2 * dusttogas * Sigmap2 - \
                np.pi * (r1 * cmperau)**2 * dusttogas * Sigmap1


def Mgas_dt(r1, r2, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0, dusttogas = 0.01, \
             Sigmad = 0, Sigmad1 = 0, Sigmad2 = 0):
     if Sigmad == 0:
         return np.pi * (r2 * cmperau)**2 * Sigmadisk(r2, Sigma0, betaS) - \
                np.pi * (r1 * cmperau)**2 * Sigmadisk(r1, Sigma0, betaS)
     else:
         return np.pi * (r2 * cmperau)**2 * Sigmad2 - \
                np.pi * (r1 * cmperau)**2 * Sigmad1




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
     


def n_no_acc(rin, sin, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0):

     rfin = rf(rin, sin, mx, Ex, Nx, rhos, T0, betaT, mu, Sigma0, betaS, \
         Mstar, sigma, npts, nptsin, tin, eps, vphi)

     t, a, s = rfin

     ngas = 1 - s**3 / sin**3

     return ngas, a


def n_with_acc(rin, sin, mx, Ex, Mdotgas, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, eps = 0, vphi = 0, dusttogas = 0.01):

     rfin = rf(rin, sin, mx, Ex, Nx, rhos, T0, betaT, mu, Sigma0, betaS, \
         Mstar, sigma, npts, nptsin, tin, eps, vphi)

     t, a, s = rfin

     nsol_noacc = s**3 / sin**3
     ngas = 1 - s**3 / sin**3

     Mdotsol = []
     for i in range(len(a)):
          Mdotsol = np.append(Mdotsol, Mdot_solids(a[i], s[i], mx, Ex, Nx, rhos, T0, betaT, mu, Sigma0, betaS, \
              Mstar, sigma, npts, nptsin, tin, eps, vphi))

     #Mdotsol = Mdot_solids(a, sin, mx, Ex, Nx, rhos, T0, betaT, mu, Sigma0, betaS, \
     #    Mstar, sigma, npts, nptsin, tin, eps, vphi)

     ngas = 1 - Mdotsol / Mdotgas
     

     return ngas, a, Mdotsol



     




        
        
