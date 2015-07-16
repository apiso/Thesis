from utils.constants import G, kb, mp, Msun, cmperau, AU
import numpy as np
from numpy import ones, zeros, shape,arange,Inf,maximum,minimum,exp,array,sqrt,invert
from T_freeze import T_freeze, Rdes, tevap, vib_freq
from C_to_O import T_freeze_H20, T_freeze_CO2, T_freeze_CO
import scipy
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from utils.zbrac import zbrac
from utils.utilities import tridag
#from advection_from_Til.two-pop-py.src

###################################################################

def impl_donorcell_adv_diff_delta(n_x,x,Diff,v,g,h,K,L,flim,u_in,dt,pl,pr,ql,qr,rl,rr,coagulation_method,A,B,C,D):
    """
    Implicit donor cell advection-diffusion scheme with piecewise constant values
    
        Perform one time step for the following PDE:
    
           du    d  /    \    d  /              d  /       u   \ \
           -- + -- | u v | - -- | h(x) Diff(x) -- | g(x) ----  | | = K + L u
           dt   dx \    /    dx \              dx \      h(x) / /
    
        with boundary conditions
    
            dgu/h |            |
          p ----- |      + q u |       = r
             dx   |x=xbc       |x=xbc
    Arguments:
          n_x   = # of grid points
          x     = the grid
          Diff  = value of Diff @ cell center
          v     = the values for v @ interface (array[i] = value @ i-1/2)
          g     = the values for g(x)
          h     = the values for h(x)
          K     = the values for K(x)
          L     = the values for L(x)
          flim  = diffusion flux limiting factor at interfaces
          u     = the current values of u(x)
          dt    = the time step
    
    OUTPUT:
          u     = the updated values of u(x) after timestep dt
    
    """
    D05=zeros(n_x)
    h05=zeros(n_x)
    rhs=zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    for i in arange(1,n_x):
        D05[i] = flim[i] * 0.5 * (Diff[i-1] + Diff[i])
        h05[i] = 0.5 * (h[i-1] + h[i])
    #
    # calculate the entries of the tridiagonal matrix
    #
    for i in arange(1,n_x-1):
        vol = 0.5*(x[i+1]-x[i-1])
        A[i] = -dt/vol *  \
            ( \
            + max(0.,v[i])  \
            + D05[i] * h05[i] * g[i-1] / (  (x[i]-x[i-1]) * h[i-1]  ) \
            )
        B[i] = 1. - dt*L[i] + dt/vol * \
            ( \
            + max(0.,v[i+1])   \
            - min(0.,v[i])  \
            + D05[i+1] * h05[i+1] * g[i]   / (  (x[i+1]-x[i]) * h[i]    ) \
            + D05[i]   * h05[i]   * g[i]   / (  (x[i]-x[i-1]) * h[i]    ) \
            )
        C[i] = dt/vol *  \
            ( \
            + min(0.,v[i+1])  \
            - D05[i+1] * h05[i+1]  * g[i+1] / (  (x[i+1]-x[i]) * h[i+1]  ) \
            )
        D[i] = -dt * K[i]
    #
    # boundary Conditions
    #
    A[0]   = 0.
    B[0]   = ql - pl*g[0] / (h[0]*(x[1]-x[0]))
    C[0]   =      pl*g[1] / (h[1]*(x[1]-x[0]))
    D[0]   = u_in[0]-rl
    
    A[-1] =    - pr*g[-2] / (h[-2]*(x[-1]-x[-2]))
    B[-1] = qr + pr*g[-1]  / (h[-1]*(x[-1]-x[-2]))
    C[-1] = 0.
    D[-1] = u_in[-1]-rr
    #
    # if coagulation_method==2,
    #  then we change the arrays and
    #  give them back to the calling routine
    # otherwise, we solve the equation
    #
    if  coagulation_method==2:  
        A = A/dt           ##ok<NASGU>
        B = (B - 1.)/dt   ##ok<NASGU>
        C = C/dt           ##ok<NASGU>
        D = D/dt           ##ok<NASGU>
    else:
        #
        # the old way
        #
        #rhs = u - D
    
        #
        # the delta-way
        #
        for i in arange(1,n_x-1):
            rhs[i] = u_in[i] - D[i] - (A[i]*u_in[i-1]+B[i]*u_in[i]+C[i]*u_in[i+1])
        rhs[0]   = rl - (               B[0] *u_in[0]  + C[0]*u_in[1])
        rhs[-1]  = rr - (A[-1]*u_in[-2]+B[-1]*u_in[-1]               )
        #
        # solve for u2
        #
        u2=tridag(A,B,C,rhs,n_x);
        #
        # update u
        #u = u2   # old way
        #
        u_out = u_in+u2 # delta way
    
    return u_out

###########################


def Tdisk(r, T0 = 120, betaT = 3./7):
    
    """
    
    Calculates the disk temperature as a function of semimajor axis for a passive disk.
    
    Input
    -----
    r:
        semimajor axis in AU
    T0:
        temperature at 1 AU in K
    betaT:
        powerlaw coefficient in T = T0 * r**(-betaT)
        
    Output
    ------
    disk temperature in K
    
    """
    
    return ((T0 * r**(-betaT))**4 + 10000.)**0.25
     
def gamma(betaT = 3./7):
    
    """
    Powerlaw coefficient in the viscosity dependendce,
    nu \propto r^gamma
    
    Input
    -----
    betaT:
        powerlaw coefficient in the temperature dependence
        
    Output
    ------
    gamma (dimensionless)
    
    """
     
    return -betaT + 3./2

def cdisk(r, T0 = 120, betaT = 3./7, mu = 2.35):
    
    """
    Sound speed as a function of semimajor axis
    
    Input
    -----
    r:
        semimajor axis in AU
    T0, betaT:
        normalization temperature and powerlaw in the disk temperature dependence
    mu:
        mean molecular weight of the nebular gas
        
    Output
    ------
    sound speed in cm s^-1
           
    """
    
    return np.sqrt(kb * Tdisk(r, T0, betaT) / (mu * mp))

def vth(r, T0 = 120, betaT = 3./7, mu = 2.35):
    
    """
    Mean thermal velocity for a Maxwellian distribution
    
        Input
    -----
    r:
        semimajor axis in AU
    T0, betaT:
        normalization temperature and powerlaw in the disk temperature dependence
    mu:
        mean molecular weight of the nebular gas (dimensionless)
        
    Output
    ------
    thermal velocity in cm s^-1
    
    """
    return np.sqrt(8 / np.pi) * cdisk(r, T0, betaT, mu)
    
def Omegak(r, Mstar = Msun):
    
    """
    
    Keplerian angular frequency
    
    Input
    -----
    r:
        semimajor axis in AU
    Mstar:
        host star mass in g
        
    Output
    ------
    Omegak in s^-1
    
    """
    return np.sqrt(G * Mstar / (r * cmperau)**3)

def Hdisk(r, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun):
    
    """
    
    Disk scale height
    
    Input
    -----
    r:
        semimajor axis in AU
    T0, betaT:
        normalization temperature and powerlaw in the disk temperature dependence
    mu:
        mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
        
    Output
    ------
    disk scale height in cm
    
    """
    return cdisk(r, T0, betaT, mu) / Omegak(r, Mstar)


def nu(r, alpha, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun):
    
    """
    
    Kinematic viscosity
    
    Input
    -----
    r:
        semimajor axis in AU
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    T0, betaT:
        normalization temperature and powerlaw in the disk temperature dependence
    mu:
        mean molecular weight (dimensionless)
    Mstar:
        host star mass in g       
    
    Output
    ------
    viscosity in cm^2 s^-1
    
    """

    return alpha * cdisk(r, T0, betaT, mu) * Hdisk(r, T0, betaT, mu, Mstar)
        

def Sigmadisk(r, t, alpha, Mdisk, rc, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, gammadflag = 0):
    
     """
     
     Gas surface density
     
     Input
     -----
     r:
         semimajor axis in AU
     t:
         time at which sigma is calculated in s
     alpha:
         coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
     Mdisk:
         total disk mass in g
     rc:
         characteristic disk radius at which the surface density starts decreasing exponentially in cm
     T0, betaT:
         normalization temperature and powerlaw in the disk temperature dependence
     mu:
         mean molecular weight of the gas (dimensionless)
     Mstar:
         host star mass in g
     gammadflag:
         flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
         if set to 1, gamma = 1, as is the case in "textbook" examples
         
     Output
     ------
     gas surface density in g cm^-2
           
     """    

     if gammadflag == 0:
         gammad = gamma(betaT)
     else:
         gammad = 1
         
     nu1 = nu(rc / cmperau, alpha, T0, betaT, mu, Mstar)
     ts = 1. / (3 * (2 - gammad)**2) * rc**2 / nu1
     T = t / ts + 1
     rtild = r * cmperau / rc

     if r >= 4:
        return Mdisk / (2 * np.pi * rc**2 * rtild**gammad) * T**(- (2.5 - gammad) / (2 - gammad)) \
            * np.exp(- rtild**(2 - gammad) / T)
     else:
         return Mdisk / (2 * np.pi * rc**2 * rtild**gammad) * T**(- (2.5 - gammad) / (2 - gammad)) \
            * np.exp(- rtild**(2 - gammad) / T) / 1000
                

def vacc_act(r, t, alpha, mu = 2.35, T0 = 120, betaT = 3./7, rc = 100 * AU, Mstar = Msun, gammadflag = 0):
    
    """
    
    Accretion velocity of the disk gas for an active disk
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time at which the accretion velocity is calculated in s
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    mu:
        mean molecular weight of the gas (dimensionless)
    T0, betaT:
        temperature normalization and powerlaw coefficient in the disk temperature profile
    rc:
        characteristic disk radius in cm
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    
    Output
    ------
    gas accretion velocity in cm s^-1  
                           
    """
    
    if gammadflag == 0:
        gammad = gamma(betaT)
    else:
        gammad = 1
    nu1 = nu(rc / cmperau, alpha, T0, betaT, mu, Mstar)
    ts = 1. / (3 * (2 - gammad)**2) * rc**2 / nu1
    T = t / ts + 1
    rtild = r * cmperau / rc

    return (3*alpha*kb*r**(-4*betaT)*rtild**(-gammad-1) * (rc**2*T*(1e4*(gammad-2)*r**(4*betaT)+T0**4*(betaT+gammad-2))*\
            rtild**gammad-cmperau**2*(gammad-2)*r**2*(1e4*r**(4*betaT)+T0**4))) / \
                (mp*mu*rc**3*T*(T0**4*r**(-4*betaT)+1e4)**(3./4)*np.sqrt(G*Mstar/(r**3*cmperau**3)))    

    
def rhodisk(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0):
    
    """
    
    Disk gas density
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time at which density is calculated in s
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    Mdisk:
        disk mass in g
    rc:
        characteristic disk radius in cm
    T0, betaT:
        temperature normalization and powerlaw coefficient in disk temperature dependence
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
        
    Output
    ------
    disk density in g cm^-3
        
    """
    
    return Sigmadisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) / \
            (np.sqrt(2 * np.pi) * Hdisk(r, T0, betaT, mu, Mstar))
    
def Pdisk(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0):
    
    """
    
    Disk pressure
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time at which density is calculated in s
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    Mdisk:
        disk mass in g
    rc:
        characteristic disk radius in cm
    T0, betaT:
        temperature normalization and powerlaw coefficient in disk temperature dependence
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
        
    Output
    ------
    disk pressure in dyne
        
    """
    
    return rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * cdisk(r, T0, betaT, mu)**2

        
def lambdamfp(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0, sigma = 2e-15):
    
    """
    
    Gas mean free path
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2
        
    Output
    ------
    mfp in cm
    
    """

    return 1 / (np.sqrt(2) * sigma * \
          (rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) / (mu * mp)))

def eta(r, t, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0, sigma = 2e-15):
    
    """
    
    (dP/dr) * 1 / (2 * rhodisk * vk**2)
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    eta (dimensionless)
       
    """
    
    return - (r * AU) / (2 * rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * (r * AU * Omegak(r, Mstar))**2) \
            * (Pdisk(r+dr, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) - \
                    Pdisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag)) / (dr * AU)


    return  Pdisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) / \
        (2 * rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * (r * AU * Omegak(r, Mstar))**2) * 2.75       
                     

def ts(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    
    """
    Stopping time due to laminar gas drag for a particle of size s
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    rhos:
        density of the solid particle of size s in g cm^-3
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    stopping time in s    
    
    """

    if s <= 9 * lambdamfp(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) / 4:

        return rhos * s / (rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * \
            vth(r, T0, betaT, mu)) #Epstein drag regime
            
    else: #Stokes drag regime with a prescription dependent on the Reynolds number

        def f(t):
             
        
             vroverr = -2 * eta(r, t, alpha, dr, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar) * \
                       (t* Omegak(r, Mstar)) / (1 + t * Omegak(r, Mstar))
             vphioverr = - eta(r, t, alpha, dr, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar) * \
                    (1. / (1 + (t * Omegak(r, Mstar))**2) - 1)
             vrel = np.sqrt(vroverr**2 + vphioverr**2)
     
             Re = 4 * vrel * s / \
                  (lambdamfp(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) * vth(r, T0, betaT, mu))
     
             CD = 24.0/Re * (1.0+0.27*Re)**0.43 + 0.47 * (1.0 - np.exp(-0.04 * Re**0.38))
        

             return 8 * rhos * s / (3 * rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * \
                      vrel * CD) - t

        time = brentq(f, 1e-20, 1e30)

        return time
          
        
def taus(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    
    """
    Stokes number (or dimensionless stopping time) = ts * Omegak
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    rhos:
        density of the solid particle of size s in g cm^-3
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    Stokes number (dimensionless)    
    
    """
    
    return ts(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar) 

        
def tr(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):

    """
    Radial drift timescale due to laminar gas drag for a particle of size s
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    rhos:
        density of the solid particle of size s in g cm^-3
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    radial drift time in s    
    
    """

    return (1 + taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2)/ \
           taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) / \
           (2 * eta(r, t, alpha, dr, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar))


def rdot(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    
    """
    Radial drift velocity due to laminar gas drag for a particle of size s
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    rhos:
        density of the solid particle of size s in g cm^-3
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    drift velocity in cm s^-1    
    
    """

    return - (r * cmperau) / tr(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)
    

#def t_gas_acc(r, alpha, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2, Sigmad = 0):
#     return 1. / (2 * alpha * Omegak(r, Mstar) * eta(r, T0, betaT, mu, Mstar, betaS, Sigmad))

def rdot_with_acc(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    
    """
    Radial drift velocity when the gas accretion velocity is taken into account
    (which is the case for an active disk)
    
    Input
    -----
    r:
        semimajor axis in AU
    t:
        time in s
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    dr:
        differential dr in AU
    Mdisk:
        disk mass in g
    rc:
        disk characteristic radius in cm
    T0, betaT:
        normalization temperature and powerlaw coefficient in the disk temperature dependence
    rhos:
        density of the solid particle of size s in g cm^-3
    mu:
        gas mean molecular weight (dimensionless)
    Mstar:
        host star mass in g
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma:
        cross section for collisions in cm^2 
        
    Output
    ------
    radial drift velocity with gas accretion in cm s^-1    
    
    """

    return rdot(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) + \
        vacc_act(r, t, alpha, mu, T0, betaT, rc, Mstar, gammadflag) / \
              (1 + taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2)

#######################################################################################

def tdes(mx, Ex, Tx, s, Nx = 1e15, rhos = 3.0):
    
     """
     Desorption time for a volatile particle of size s and composed of single molecular
     species (e.g., H2O, CO2, CO)
     
     Input
     -----
     mx:
         molecular weight of the volatile (dimensionless)
     Ex:
         binding energy in K
     Tx:
         freezing temperature of the volatile in K
         
     s:
         size of the solid particle in cm
     Nx:
         number of adsorption sites per cm^2
     rhos:
         density of solid particle in g cm^-3
         
     Output
     ------
     desorption time in s 
     
     """
     
     return rhos / (3 * mx * mp) * s / (Nx * Rdes(mx, Ex, Tx)) 


def r_freeze(mx, Ex, nx, T0 = 120., betaT = 3./7):
    
     """
     Snowline location for a given species and disk temperature profile
     
     Input
     -----
     mx:
         mean molecular weight of the volatile (dimensionless)
     Ex:
         binding energy in K
     nx:
         number density of the volatile in the disk midplane
     T0, betaT:
         normalization temperature and powerlaw coefficient in the disk temperature profile
         
     Output
     ------
     snowline radius in AU
     
     """

     def f(x):
          return Tdisk(x, T0, betaT) - T_freeze(mx, Ex, nx)

     return brentq(f, 0.1, 100)
     

def r_stop(t, s, alpha, dr, Mdisk, mx, Ex, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15), Nx = 1e15):

     def f(r):
          return (r*AU) / (- rdot_with_acc(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
            Mstar, gammadflag, sigma)) - tdes(mx, Ex, Tdisk(r, T0, betaT), s, Nx, rhos)

     try:
          return brentq(f, 1e-3, 1e2)
     except ValueError:
          return 1e-10


def rf(rin, tf, sin, mx, Ex, alpha, dr, Mdisk, rc = 100*AU, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 1e6, nptsin = 1e4, tin = 1e-10, gammadflag = 0, returnall = False):

     def f(x, t):
          
          return np.array([ \
               rdot_with_acc(x[0] / cmperau, t, x[1], alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma), \
               - 3 * mx * mp  / rhos * \
                    Nx * Rdes(mx, Ex, Tdisk(x[0] / cmperau, T0, betaT))])
          

     tv = np.logspace(np.log10(tin), np.log10(tf), npts)
     y = odeint(f, [rin * cmperau, sin], tv)


     for i in range(len(y[:,0]) - 1):
          if y[:,1][i] >=0 and y[:,1][i + 1] < 0:
               break
          #sf = y[:,1][i - 1]
          #npts = 10 * npts
     if i == len(y[:,0] - 1):
        #print "No desorption." 
        if returnall == True:
          return tv, y[:,0] / cmperau, y[:,1]
        else:
          return y[:,0][-1] / cmperau
     else:
	  tint = np.logspace(np.log10(tv[i]), np.log10(tv[i + 1]), nptsin)
	  yint = odeint(f, [y[:,0][i], y[:,1][i]], tint)

     	  finta = interp1d(yint[:,1][::-1], yint[:,0][::-1])
     	  fintt = interp1d(yint[:,1][::-1], tint)

     	  try:
     	      #print "It's desorbing."
     	      #print i
              af = float(finta(0))
              tf = float(fintt(0))

              afv = np.append(y[:,0][:i], af)
              sfv = np.append(y[:,1][:i], 0)
              tfv = np.append(tv[:i], tf)
              
              if returnall == True:
                  return tfv, afv / cmperau, sfv
              else:
     	          return float(finta(0)) / cmperau
          except ValueError:
              #print "It's desorbing but there's an issue."
              #print i
              if returnall == True:
                  return tv, y[:,0] / cmperau, y[:,1]
              else:
                  return y[:,0][-1] / cmperau


def rdrift(rin, tin, tf, s, alpha, dr, Mdisk, rc = 100*AU, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), npts = 100, gammadflag = 0):
    
    def f(x, t):
          
        return rdot_with_acc(x / cmperau, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)
          

    t = np.logspace(np.log10(tin), np.log10(tf), npts)
    y = odeint(f, rin * cmperau, t)
    
    return y[-1][0] / AU
    
    


#######################################################################################



def Sigmap_act(rin, rout, nr, ti, tf, nt, s, alpha, Mdisk, rc, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0, sigmad_dt = 1, dif = 1, lin = 1):
    
    """
    Surface density of solids evolved in time using the advection-diffusion equation
    
    Input
    -----
    rin:
        starting point of the radial grid size in AU
    rout:
        ending point of the radial grid size in AU
    nr:
        number of radial grid points
    ti:
        time at which we begin the evolution in s
    tf:
        time at which we end the evolution in s
    nt:
        number of time steps
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    Mdisk:
        disk mass in g
    rc:
        characteristic disk radius in cm
    rhos:
        density of the solid particle in g cm^-3
    T0, betaT:
        temperature normalization and powerlaw coefficient in the disk temperature profile
    mu:
        mean molecular weight of the disk gas (dimensionless)
    Mstar:
        host star mass in g
    sigma:
        cross section for collisions in cm^2
    dusttogas:
        dust-to-gas ratio in the disk (dimensionless)
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma_dt:
        flag; if sigma_dt = 1, we allow the gas surface density to evolve in time;
        otherwise, the gas surface density does not change in time
    
    """
    
    r = np.logspace(np.log10(rin), np.log10(rout), nr) #radial grid
    
    if lin == 1:
        t = np.linspace(ti, tf, nt)
        dt = t[1] - t[0]  #temporal grid
    else:
        t = np.logspace(np.log10(ti), np.log10(tf), nt)
     #time step in the advection-diffusion equation solver; set to be constant for now
    
    v, Sigmad, D = [], [], [] #initializing arrays for radial drift velocity, gas surface density and diffusivity
    
    sigarray = np.ndarray(shape = (nr, nt), dtype = float) #initializing array for the dust surface density at each time step
    
       
    for i in range(nr):
        
        #ensuring that we don't run into an index error when setting up dr 
        if i != nr - 1:
            dr = r[i + 1] - r[i]
        else:
            dr = r[-1] - r[-2]
        
        #setting up v, Sigmad and D at time zero        
        v = np.append(v, rdot_with_acc(r[i], t[0], s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
                Mstar, gammadflag, sigma))
        Sigmad = np.append(Sigmad, Sigmadisk(r[i], t[0], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
        
        if dif == 1:
            D = np.append(D, alpha * cdisk(r[i], T0, betaT, mu) * Hdisk(r[i], T0, betaT, mu, Mstar) / \
                (1 + taus(r[i], t[0], s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2))
        else:
            D = zeros(nr)
        
    
    #setting up the solver        
    h = Sigmad * r * AU
    uin = r * Sigmad * dusttogas * AU
    
    #uin = r * maximum(Sigmad[280] / 100 * np.exp(-(r-r[280])**2/(2*(0.1)**2)), 1e-100) * AU
    
    #uin = []
    #for i in range(nr):
    #    if r[i] >= 10. and r[i] <= 10.5:
    #        uin = np.append(uin, r[i] * Sigmad[i] * dusttogas * AU)
    #    else:
    #        uin = np.append(uin, 1e-100)
    
    g = ones(nr)
    K = zeros(nr)
    L = zeros(nr)
    flim = ones(nr)
    
    A0 = zeros(nr)
    B0 = zeros(nr)
    C0 = zeros(nr)
    D0 = zeros(nr)
    
    for i in range(nr):
        #sigarray[i, 0] = Sigmad[i] * dusttogas
        sigarray[i, 0] = uin[i] / (r[i] * AU)
        
    time = t[0]
    
    for j in range(1, nt):
        if lin !=1 :
            dt = t[j] - t[j - 1]
        uout = impl_donorcell_adv_diff_delta(nr, r * AU, D, v, g, h, K, L, flim, uin, dt, 1,1, 0, 0, 0,0, 1, A0, B0, C0, D0)
        
        for i in range(nr):
            sigarray[i, j] = uout[i] / (r[i] * AU)
        
        uin = uout #updating the new dust surface density to be used in the next time step
        time = time + dt #moving on to the next time step
        
        if sigmad_dt !=0: #if sigmad_dt = 0, we skip updating v, Sigmad, D
            
            v, Sigmad, D = [], [], [] #initializing new arrays
        
        
            for i in range(nr):
                
                #updating v, Sigmad, D with their values at the new time step
                v = np.append(v, rdot_with_acc(r[i], time, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
                    Mstar, gammadflag, sigma))
                Sigmad = np.append(Sigmad, Sigmadisk(r[i], time, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
                
                if dif == 1:
                    D = np.append(D, alpha * cdisk(r[i], T0, betaT, mu) * Hdisk(r[i], T0, betaT, mu, Mstar) / \
                        (1 + taus(r[i], time, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2))
                else:
                    D = zeros(nr)
                        #D is the dust diffusivity, i.e. Dgas / (1+St^2), with Dgas the viscosity and St the Stokes number
            h = Sigmad * r * AU
        
    
    return sigarray



def Sigmadisk_cavity(rin, rout, nr, ti, tf, nt, s, alpha, Mdisk, rc, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0, sigmad_dt = 1, dif = 1, lin = 1):
    
    """
    Surface density of solids evolved in time using the advection-diffusion equation
    
    Input
    -----
    rin:
        starting point of the radial grid size in AU
    rout:
        ending point of the radial grid size in AU
    nr:
        number of radial grid points
    ti:
        time at which we begin the evolution in s
    tf:
        time at which we end the evolution in s
    nt:
        number of time steps
    s:
        particle size in cm
    alpha:
        coefficient in the Shakura-Sunyaev disk prescription (dimensionless)
    Mdisk:
        disk mass in g
    rc:
        characteristic disk radius in cm
    rhos:
        density of the solid particle in g cm^-3
    T0, betaT:
        temperature normalization and powerlaw coefficient in the disk temperature profile
    mu:
        mean molecular weight of the disk gas (dimensionless)
    Mstar:
        host star mass in g
    sigma:
        cross section for collisions in cm^2
    dusttogas:
        dust-to-gas ratio in the disk (dimensionless)
    gammadflag:
        flag: if set to 0, the gamma in the viscosity powerlaw dependence is calculated;
        if set to 1, gamma = 1, as is the case in "textbook" examples
    sigma_dt:
        flag; if sigma_dt = 1, we allow the gas surface density to evolve in time;
        otherwise, the gas surface density does not change in time
    
    """
    
    r = np.logspace(np.log10(rin), np.log10(rout), nr) #radial grid
    
    if lin == 1:
        t = np.linspace(ti, tf, nt)
        dt = t[1] - t[0]  #temporal grid
    else:
        t = np.logspace(np.log10(ti), np.log10(tf), nt)
     #time step in the advection-diffusion equation solver; set to be constant for now
    
    v, Sigmad = [], [] #initializing arrays for radial drift velocity, gas surface density and diffusivity
    
    sigarray = np.ndarray(shape = (nr, nt), dtype = float) #initializing array for the dust surface density at each time step
    
       
    for i in range(nr):
        
        #ensuring that we don't run into an index error when setting up dr 
        if i != nr - 1:
            dr = r[i + 1] - r[i]
        else:
            dr = r[-1] - r[-2]
        
        #setting up v, Sigmad and D at time zero        
        v = np.append(v, vacc_act(r[i], t[0], alpha, mu, T0, betaT, rc, Mstar, gammadflag = 0))
        Sigmad = np.append(Sigmad, Sigmadisk(r[i], t[0], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
        
        D = zeros(nr)
        
    
    #setting up the solver        
    h = Sigmad * r * AU
    uin = r * Sigmad * AU
    
    #uin = r * maximum(Sigmad[280] / 100 * np.exp(-(r-r[280])**2/(2*(0.1)**2)), 1e-100) * AU
    
    #uin = []
    #for i in range(nr):
    #    if r[i] >= 10. and r[i] <= 10.5:
    #        uin = np.append(uin, r[i] * Sigmad[i] * dusttogas * AU)
    #    else:
    #        uin = np.append(uin, 1e-100)
    
    g = ones(nr)
    K = zeros(nr)
    L = zeros(nr)
    flim = ones(nr)
    
    A0 = zeros(nr)
    B0 = zeros(nr)
    C0 = zeros(nr)
    D0 = zeros(nr)
    
    for i in range(nr):
        #sigarray[i, 0] = Sigmad[i] * dusttogas
        sigarray[i, 0] = uin[i] / (r[i] * AU)
        
    time = t[0]
    
    for j in range(1, nt):
        if lin !=1 :
            dt = t[j] - t[j - 1]
        uout = impl_donorcell_adv_diff_delta(nr, r * AU, D, v, g, h, K, L, flim, uin, dt, 1,1, 0, 0, 0,0, 1, A0, B0, C0, D0)
        
        for i in range(nr):
            sigarray[i, j] = uout[i] / (r[i] * AU)
        
        uin = uout #updating the new dust surface density to be used in the next time step
        time = time + dt #moving on to the next time step
        
        if sigmad_dt !=0: #if sigmad_dt = 0, we skip updating v, Sigmad, D
            
            v, Sigmad = [], [] #initializing new arrays
        
        
            for i in range(nr):
                
                #updating v, Sigmad, D with their values at the new time step
                v = np.append(v, vacc_act(r[i], time, alpha, mu, T0, betaT, rc, Mstar, gammadflag = 0))
                Sigmad = np.append(Sigmad, Sigmadisk(r[i], time, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
        
                D = zeros(nr)
                
                        #D is the dust diffusivity, i.e. Dgas / (1+St^2), with Dgas the viscosity and St the Stokes number
            h = Sigmad * r * AU
        
    
    return sigarray



    
def Mdot_solids(r, t, s, alpha, Mdisk = 0.1*Msun, rc = 100*AU, dr = 1e-3, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0, sigmad_dt = 1):
    
    nt = 50
    nr = 50
    
    rgrid = np.logspace(np.log10(0.05),np.log10(4e3),nr)
    tgrid = np.linspace(1e2,3e6,nt)*365*24*3600
    
    sig = Sigmap_act(rgrid[0], rgrid[-1], nr, tgrid[0], tgrid[-1], nt, s, alpha, Mdisk, rc, rhos, T0, betaT, mu, \
        Mstar, sigma, dusttogas, gammadflag, sigmad_dt)
    
    func = scipy.interpolate.interp2d(tgrid, rgrid, sig)
    sigmap = func(t, r)    
       
    return -2 * np.pi * (r * AU) * rdot_with_acc(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
        Mstar, gammadflag, sigma) * sigmap, sigmap, sig
    

def dMdot_solids(rin, rout, t, s, alpha, Mdisk = 0.1*Msun, rc = 100*AU, dr = 1e-3, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0, sigmad_dt = 1):
    
         Mdotout, sigout, sig = Mdot_solids(rout, t, s, alpha, Mdisk, rc, dr, rhos, T0, betaT, mu, \
            Mstar, sigma, dusttogas, gammadflag, sigmad_dt)
         Mdotin, sigin, sig =   Mdot_solids(rin, t, s, alpha, Mdisk, rc, dr, rhos, T0, betaT, mu, \
                    Mstar, sigma, dusttogas, gammadflag, sigmad_dt)
         return Mdotout - Mdotin, sigin, sigout, sig  
                    
                       
def Mdot_gas(r, t, alpha, Mdisk = 0.1*Msun, mu = 2.35, T0 = 120, betaT = 3./7, rc = 100 * AU, Mstar = Msun, gammadflag = 0):
    
       return - 2 * np.pi * (r * AU) * vacc_act(r, t, alpha, mu, T0, betaT, rc, Mstar, gammadflag) * \
            Sigmadisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag)

def dMdot_gas(rin, rout, t, alpha, Mdisk = 0.1*Msun, mu = 2.35, T0 = 120, betaT = 3./7, rc = 100 * AU, Mstar = Msun, gammadflag = 0):
    
    return Mdot_gas(rout, t, alpha, Mdisk, mu, T0, betaT, rc, Mstar, gammadflag) - \
        Mdot_gas(rin, t, alpha, Mdisk, mu, T0, betaT, rc, Mstar, gammadflag)
        
 
        
               
                      
                             
                                           
        
def dMdot(rin, rout, ti, tf, nt, rstart, s, alpha, mx, Ex, dtf = 0.001, Mdisk = 0.1*Msun, rc = 100*AU, dr = 1e-3, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0, sigmad_dt = 1, f = 0.9*1e-4, Nx = 1e15, returnall = True, \
    npts = 1e6, nptsin = 1e4, tin = 1e-10, lin = 0):
    
    nr = 50
    rgrid = np.logspace(np.log10(0.05),np.log10(4e3),nr)
    if lin == 1:
        t = np.linsapce(ti, tf, nt)
    else:
        t = np.logspace(np.log10(ti),np.log10(tf),nt)
    
    sigarray = Sigmap_act(rgrid[0], rgrid[-1], nr, t[0], t[-1], nt, s, alpha, Mdisk, rc, rhos, T0, betaT, mu, \
        Mstar, sigma, dusttogas, gammadflag, sigmad_dt, lin)
        
    func = scipy.interpolate.interp2d(t, rgrid, sigarray)
    
    Ms, Mg, Mg_tot, Ms_tot, sig = [], [], [], [], []
    
    tf, af, sf = rf(rstart, tf, s, mx, Ex, alpha, dr, Mdisk, rc, Nx, rhos, T0, betaT, mu, \
            Mstar, sigma, npts, nptsin, ti, gammadflag, returnall)
    
    if sf[-1] != 0.0 or af[-1] < rin:        
    
        for i in range(nt):
            
            Mg = np.append(Mg, 0)
            
            def fint(x):
                return 2 * np.pi * func(t[i], x/AU) * x
            Ms = np.append(Ms, quad(fint, rin*AU, rout*AU)[0])
            
            def fsig(x):
                return 2 * np.pi * Sigmadisk(x/AU, t[i], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * x
            Mg_tot = np.append(Mg_tot, quad(fsig, rin*AU, rout*AU)[0])      
 
        return Mg, Ms, t, Mg_tot
        
        
        
            
    elif af[-1] >= rin:# and af[-1] <= rout:
        print "It is desorbing."
        
        sigmapdes = func(tf[-1], af[-1])
        Des = 3 * sigmapdes * ((rout*AU)**2-(rin*AU)**2) * 12*np.pi / (4 * rhos * s) * \
                mu * mp * Nx * Rdes(mx, Ex, Tdisk(af[-1], T0, betaT)) 
            
        for i in range(nt - 1):
            
            if t[i] <= tf[-1] and t[i + 1] > tf[-1]:
                break
        k = i
        print k
            
        for i in range(k + 1):

            Mg = np.append(Mg, 0)
            
            def fint(x):
                return 2 * np.pi * func(t[i], x/AU) * x
            Ms = np.append(Ms, quad(fint, rin*AU, rout*AU)[0])
            
            def fsig(x):
                return 2 * np.pi * Sigmadisk(x/AU, t[i], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * x
            Mg_tot = np.append(Mg_tot, quad(fsig, rin*AU, rout*AU)[0])             

             
             
        def fsig(x):
            return 2 * np.pi * Sigmadisk(x/AU, tf[-1], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * x
        Mg_tot_bf_des = quad(fsig, rin*AU, rout*AU)[0]
                    
        def fint(x):
            return 2 * np.pi * func(tf[-1], x/AU) * x
        Ms_bf_des = quad(fint, rin*AU, rout*AU)[0]
    
        Mdots_in = 0 
        Mdots_out = -2 * np.pi * (rin * AU) * rdot_with_acc(rin, tf[-1], s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
            Mstar, gammadflag, sigma) * func(tf[-1], rin)
        dMdots = Mdots_out - Mdots_in

        dMdotg_new = Des
        dMdots_new = dMdots - Des   
        
        Mg_at_des = dMdotg_new * (dtf*tf[-1])
        Ms_at_des = dMdots_new * (dtf*tf[-1]) + Ms_bf_des
        
        Mg_after_des, Ms_after_des = [], []
        
        for i in range(k + 1, nt):           
            Mg_after_des = np.append(Mg_after_des, Mg_at_des)
            Ms_after_des = np.append(Ms_after_des, 0)
            
            #for i in range(npts_at_des, nt - 1):
                          
        if af[-1] > rout:
            return np.append(np.append(Mg, Mg[-1]), Mg_after_des), \
            np.append(np.append(Ms, Ms[-1]), Ms_after_des), \
                np.append(np.append(t[:k + 1], tf[-1]), t[k+1:])#, \
                    #np.append(Mg_tot, Mg_tot_bf_des)  
                
        else:    
                              
            return np.append(np.append(Mg, Mg_at_des), Mg_after_des), \
                np.append(np.append(Ms, Ms_at_des), Ms_after_des), \
                    np.append(np.append(t[:k + 1], tf[-1]), t[k+1:])#, \
                       # np.append(Mg_tot, Mg_tot_bf_des)  
        
                                        

    







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
     
     
     
#def r_stop(s, mx, Ex, Nx = 1e15, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, Sigma0 = 2200, betaS = 3./2, \
#    Mstar = Msun, sigma = 2 * 10**(-15)):
#    
#     """
#     
#     Desorption distance estimated analytically by equating the desorption and 
#     drift timescales
#     
#     """
#     
#     def f(r):
#          return tr(r, s, rhos, T0, betaT, mu, Sigma0, betaS, Mstar, sigma) - \
#                 tdes(mx, Ex, Tdisk(r, T0, betaT), s, Nx, rhos)
#
#     try:
#          return brentq(f, 1e-3, 1e2)
#     except ValueError:
#          return 1e-10

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



     




        
        
