from utils.constants import G, AU, Me, Re, Msun, kb, Rfn
import numpy as np
from scipy.optimize import brentq

def Tdisk(a, b, FT):
    
    """
    Disk temperature

    Input
    ----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis dependence
    FT:
        normalization factor relative to the MMSN

    Output:
    -------
    Tdisk:
        disk temperature in K
    
    """

    return 300 * FT * a**(-b)


def RBondi(mc, a, b, FT, Y = 0.3):

    """
    Bondi radius

    Input:
    -----
    mc:
        core mass in Earth masses
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis dependence
    FT:
        normalization factor relative to the MMSN
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    Bondi radius in cm
        
    """

    return G * mc * Me / (Rfn(Y) * Tdisk(a, b, FT))


def RHill(mc, a, mstar):

    """
    Hill radius

    Input
    -----
    mc:
        core / planet mass in Earth masses
    a:
        semi-major axis in AU
    mstar:
        central star mass in Solar masses

    Output
    ------
    Hill radius in cm
    
    """

    return a * AU * (mc * Me / (3 * mstar * Msun))**(1./3)


def sigmadisk(FSigma, a, d):

    """

    gas surface density in cm^-2

    Input
    -----
    FSigma:
        normalization relative to the MMSN
    a:
        semi-major axis in AU
    d:
        power-law index in semi-major axis dependence

    Output
    ------
    Simgad

    """

    return 2 * 10**3 * FSigma * a**(-d)






def cdisk(a, b, FT, Y = 0.3):

    """

    isothermal sound speed

    Input
    -----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis dependence
    FT:
        normalization factor relative to the MMSN
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    cdisk in cm s^-1

    """

    return (Rfn(0.3) * Tdisk(a, b, FT))**(1./2)


def Omega(a, mstar):

    """

    Keplerian angular velocity

    Input
    -----
    a:
        semi-major axis in AU
    mstar:
        central star mass in solar masses

    Output
    ------
    Omega in s^-1
    
    """

    return (G * mstar * Msun / (a * AU)**3)**(1./2)


def sigmamax(a, b, FT, mstar, Y = 0.3):

    """

    maximal surface density in cm^-2 for which the disk is gravitationally stable (Q = 1)

    Input
    -----

    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis dependence
    FT:
        normalization factor relative to the MMSN
    mstar:
        central star mass in Solar masses
    Y:
        Helium mass fraction (default 0.3)


    Output
    ------
    Simgamax in cm^-2

    """

    return cdisk(a, b, FT, Y) * Omega(a, mstar) / (np.pi * G)


def Hdisk(a, b, FT, mstar, Y = 0.3):

    """

    disk scale height

    Input
    -----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis dependence
    FT:
        normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    disk scale height in cm

    """
    
    return cdisk(a, b, FT, Y) / Omega(a, mstar)


def rhodisk(a, b, d, FT, FSigma, mstar, Y = 0.3, maxQ = 0):

    """

    disk mid-plane density

    Input
    -----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    d:
        power-law index in semi-major axis surface density dependence
    FT:
        temperature normalization factor relative to the MMSN
    FSigma:
        surface density normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    rhodisk in g cm^-3


    """

    if maxQ == 0:
        return sigmadisk(FSigma, a, d) / (2 * Hdisk(a, b, FT, mstar, Y))
    else:
        return sigmamax(a, b, FT, mstar, Y) / (np.sqrt(2 * np.pi) * Hdisk(a, b, FT, mstar, Y))


def mtrans(a, b, FT, mstar, Y = 0.3):

    """

    mass for which RBondi = RHill

    Input
    -----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    FT:
        temperature normalization factor relative to the MMSN

    mstar:
        central star mass in Solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    mtrans in g
    
    """

    def f(mc):
        return RBondi(mc, a, b, FT, Y) - RHill(mc, a, mstar)
    
    return brentq(f, 1e-10, 1000) * Me


def mthermal(a, b, FT, mstar, Y = 0.3):

    """

    mass for which RHill = Hdisk

    Input
    -----
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    FT:
        temperature normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    mthermal in g
    
    """

    def f(mc):
        return RHill(mc, a, mstar) - Hdisk(a, b, FT, mstar, Y)
    
    return brentq(f, 1e-10, 1000) * Me
    
    

def MBondi(mc, a, b, d, FT, FSigma, mstar, Y = 0.3, maxQ = 0):

    """

    atmosphere mass inside Bondi radius as a fraction of the core mass

    Input
    -----
    mc:
        core mass in Earth masses
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    d:
        power-law index in semi-major axis surface density dependence
    FT:
        temperature normalization factor relative to the MMSN
    FSigma:
        surface density normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    MBondi in units of core masses

    """

    return 4 * np.pi * \
           rhodisk(a, b, d, FT, FSigma, mstar, Y, maxQ) * RBondi(mc, a, b, FT, Y)**3 / (mc * Me)


def MHill(a, b, d, FT, FSigma, mstar, Y = 0.3, maxQ = 0):

    """

    atmosphere mass inside Hill radius as a fraction of core mass

    Input
    -----
    mc:
        core mass in Earth masses
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    d:
        power-law index in semi-major axis surface density dependence
    FT:
        temperature normalization factor relative to the MMSN
    FSigma:
        surface density normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    MHill in units of core mass

    """

    return 4 * np.pi * \
           rhodisk(a, b, d, FT, FSigma, mstar, Y, maxQ) * (a * AU)**3 / (3 * mstar * Msun)



#---------------------------------------------------------------------------------------

def Matm(mc, a, b, d, FT, FSigma, mstar, Y = 0.3, maxQ = 0):

    """

    atmosphere mass 

    Input
    -----
    mc:
        core mass in Earth masses
    a:
        semi-major axis in AU
    b:
        power-law index in semi-major axis temperature dependence
    d:
        power-law index in semi-major axis surface density dependence
    FT:
        temperature normalization factor relative to the MMSN
    FSigma:
        surface density normalization factor relative to the MMSN
    mstar:
        central star mass in solar masses
    Y:
        Helium mass fraction (default 0.3)

    Output
    ------
    Matm in units of core masses

    """

    if RBondi(mc, a, b, FT, Y) <= RHill(mc, a, mstar):
        return MBondi(mc, a, b, d, FT, FSigma, mstar, Y, maxQ)
    else:
        return MHill(a, b, d, FT, FSigma, mstar, Y, maxQ)

#----------------------------------------------------------------------------------------------

## attempt to find mass-radius relationship for atmosphere from basic principles

def Ratm(Mc, rhoc, f, T, kappa):

    Rc = (3 * Mc / (4 * np.pi * rhoc))**(1./3)

    def func(Ratm):
        return 4 * np.pi * G * (f + 1) * Ratm**3 / (Rfn(0.3) * T * kappa * (Rc + Ratm)**2) - f

    return brentq(func, 1e-10, 1e20)
    
    












    
