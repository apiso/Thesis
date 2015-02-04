from utils.constants import G, kb, mp, Msun, cmperau, AU
import numpy as np
from numpy import ones, zeros, shape,arange,Inf,maximum,minimum,exp,array,sqrt,invert
from T_freeze import T_freeze, Rdes, tevap, vib_freq
from C_to_O import T_freeze_H20, T_freeze_CO2, T_freeze_CO
from scipy.integrate import odeint
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
     """Disk temperature in K with r in AU"""
     return ((T0 * r**(-betaT))**4 + 10000.)**0.25
     
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
     
#def Sigmadisk(r, Mdisk, rc):
#    
#   return maximum(Mdisk/(2*np.pi*rc**2)*(rc/(r*AU))*exp(-(r*AU)/rc),1e-100)
    

def Sigmadisk(r, t, alpha, Mdisk, rc, T0 = 120, betaT = 3./7, mu = 2.35, Mstar = Msun, gammadflag = 0):

     if gammadflag == 0:
         gammad = gamma(betaT)
     else:
         gammad = 1
         
     nu1 = nu(rc / cmperau, alpha, T0, betaT, mu, Mstar)
     ts = 1. / (3 * (2 - gammad)**2) * rc**2 / nu1
     T = t / ts + 1
     rtild = r * cmperau / rc

     return Mdisk / (2 * np.pi * rc**2 * rtild**gammad) * T**(- (2.5 - gammad) / (2 - gammad)) \
            * np.exp(- rtild**(2 - gammad) / T)
                

def vacc_act(r, t, alpha, mu = 2.35, T0 = 120, betaT = 3./7, rc = 100 * AU, Mstar = Msun, gammadflag = 0):
    
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
    
    
    #return -3.0*alpha*kb*Tdisk(r, Tstar, Rstar)/mu/mp/2./sqrt(G*Mstar/(r*AU))*(1.+7./4.)
    

    
def rhodisk(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0):
    """Disk gas density in g cm^-3 with r in AU"""
    return Sigmadisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) / \
            (np.sqrt(2 * np.pi) * Hdisk(r, T0, betaT, mu, Mstar))
    
def Pdisk(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0):
    return rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * cdisk(r, T0, betaT, mu)**2

        
def lambdamfp(r, t, alpha, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0, sigma = 2e-15):
    """Mean free path in cm with r in AU"""

    return 1 / (np.sqrt(2) * sigma * \
          (rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) / (mu * mp)))

def eta(r, t, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, mu = 2.35, Mstar = Msun, gammadflag=0, sigma = 2e-15):
    """Power law coefficient in P \propto r^(-n)"""
    return - (r * AU) / (2 * rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * (r * AU * Omegak(r, Mstar))**2) \
            * (Pdisk(r+dr, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) - \
                    Pdisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag)) / (dr * AU)
       
            
#def eta(r, Mdisk, Tstar, Rstar, dr, rc, mu = 2.35, Mstar = Msun):
#     """Dimensionless correction coefficient eta = vk - vgas"""
#     return n(r, Mdisk, Rstar, Tstar, rc, mu, Mstar, dr) * cdisk(r, Tstar, Rstar, mu)**2 / \
#        (2 * (Omegak(r, Mstar) * r * cmperau)**2)  

def ts(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    """Stopping time in seconds with r in AU and s in cm"""


    if s <= 9 * lambdamfp(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) / 4:

        return rhos * s / (rhodisk(r, t, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag) * \
            vth(r, T0, betaT, mu))
            
    else:

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
    """Dimensionless stopping time with r in AU and s in cm"""
    return ts(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar) 

        
def tr(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):
    """Radial drift time in seconds with r in AU and s in cm"""
    return (1 + taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2)/ \
           taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) / \
           (2 * eta(r, t, alpha, dr, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag, sigma) * Omegak(r, Mstar))


def rdot(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):

     return - (r * cmperau) / tr(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)

def t_gas_acc(r, alpha, T0 = 120., betaT = 3./7, mu = 2.35, Mstar = Msun, betaS = 3./2, Sigmad = 0):
     return 1. / (2 * alpha * Omegak(r, Mstar) * eta(r, T0, betaT, mu, Mstar, betaS, Sigmad))

#def rdot_gas(r, s, Mdotgas, Sigma0 = 2200, betaS = 3./2):
#
#     return - Mdotgas * Msun / (365 * 24 * 3600) / (Sigmadisk(r, Sigma0, betaS) * 2 * np.pi * (r * cmperau))

def rdot_with_acc(r, t, s, alpha, dr, Mdisk, rc = 100*AU, T0=120, betaT=3./7, rhos = 3.0, mu = 2.35, \
    Mstar = Msun, gammadflag = 0, sigma = 2 * 10**(-15)):

     return rdot(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma) + \
        vacc_act(r, t, alpha, mu, T0, betaT, rc, Mstar, gammadflag) / \
              (1 + taus(r, t, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, Mstar, gammadflag, sigma)**2)

#######################################################################################



def Sigmap_act(rin, rout, nr, ti, tf, nt, s, alpha, Mdisk, rc, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, gammadflag = 0):
    
    r = np.logspace(np.log10(rin), np.log10(rout), nr)
    t = np.linspace(ti, tf, nt)
    dt = t[1] - t[0]
    #dr = r[1] - r[0]
    
    v, Sigmad, D = [], [], []
    #v = np.ndarray(shape = (nr, nt + 1), dtype = float)
    #Sigmad = np.ndarray(shape = (nr, nt + 1), dtype = float)
    #D = np.ndarray(shape = (nr, nt + 1), dtype = float)
    
    sigarray = np.ndarray(shape = (nr, nt + 1), dtype = float)
    
        
    for i in range(nr):
        if i != nr - 1:
            dr = r[i + 1] - r[i]
        else:
            dr = r[-1] - r[-2]
            
        v = np.append(v, rdot_with_acc(r[i], t[0], s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
                Mstar, gammadflag, sigma))
        Sigmad = np.append(Sigmad, Sigmadisk(r[i], t[0], alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
        D = np.append(D, alpha * cdisk(r[i], T0, betaT, mu) * Hdisk(r, T0, betaT, mu, Mstar))
        
    h = Sigmad * r * AU
    uin = r * Sigmad * dusttogas * AU
    
    g = ones(nr)
    K = zeros(nr)
    L = zeros(nr)
    flim = ones(nr)
    
    A0 = zeros(nr)
    B0 = zeros(nr)
    C0 = zeros(nr)
    D0 = zeros(nr)
    
    for i in range(nr):
        sigarray[i, 0] = Sigmad[i] * dusttogas
    
    time = t[0]
    
    for j in range(nt):
        uout = impl_donorcell_adv_diff_delta(nr, r * AU, D, v, g, h, K, L, flim, uin, dt, 1,0, 0, 1, 0, r[-1]*1e-100, 1, A0, B0, C0, D0)
        
        for i in range(nr):
            sigarray[i, j + 1] = uout[i] / (r[i] * AU)
        
        uin = uout
        time = time + dt
        
        
        v, Sigmad, D = [], [], []
        
        for i in range(nr):
            
            v = np.append(v, rdot_with_acc(r[i], time, s, alpha, dr, Mdisk, rc, T0, betaT, rhos, mu, \
                Mstar, gammadflag, sigma))
            Sigmad = np.append(Sigmad, Sigmadisk(r[i], time, alpha, Mdisk, rc, T0, betaT, mu, Mstar, gammadflag))
            D = np.append(D, alpha * cdisk(r[i], T0, betaT, mu) * Hdisk(r, T0, betaT, mu, Mstar))
        
        h = Sigmad * r * AU
    
    return sigarray


#def Sigmap_act(rin, rout, nr, t, dt, s, alpha, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
#    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01):
#    
#    r = np.logspace(np.log10(rin), np.log10(rout), nr)
#    
#    v, Sigmad, D = [], [], []
#    
#    for i in range(nr):
#        v = np.append(v, rdot_with_acc(r[i], t, s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma))
#        Sigmad = np.append(Sigmad, Sigmadisk_act(r[i], t, alpha, T0, betaT, mu, Mstar, r1, C))
#        D = np.append(D, alpha * cdisk(r[i], T0, betaT, mu) * Hdisk(r[i], T0, betaT, mu, Mstar))
#        
#    h = Sigmad * r
#    uin = r * Sigmad * dusttogas
#    
#    g = ones(nr)
#    K = zeros(nr)
#    L = zeros(nr)
#    flim = ones(nr)
#    
#    A0 = zeros(nr)
#    B0 = zeros(nr)
#    C0 = zeros(nr)
#    D0 = zeros(nr)
#    
#    
#    uout = impl_donorcell_adv_diff_delta(nr, r * AU, D, v, g, h, K, L, flim, uin, dt, 1,1, 0, 0, 0, 0, 1, A0, B0, C0, D0)
#    
#    return uout / r
    
 




def dMdot(rin, rout, s, alpha, t0, tmax, n, rhos = 3.0, T0 = 120, betaT = 3./7, mu = 2.35, r1 = 100 * cmperau, C = 4.45e18, \
    Mstar = Msun, sigma = 2 * 10**(-15), dusttogas = 0.01, f = 1e-4, vrw = 0):
     
     Sigmap_in = Sigmadisk_act(rin, t0, alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas
     Sigmap_out = Sigmadisk_act(rout, t0, alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas
     
     Sigmag_in = Sigmadisk_act(rin, t0, alpha, T0, betaT, mu, Mstar, r1, C)
     Sigmag_out = Sigmadisk_act(rout, t0, alpha, T0, betaT, mu, Mstar, r1, C)

     t = np.linspace(t0, tmax, n)
     #t = np.logspace(np.log10(t0), np.log10(tmax), n)
     dM_gas, dM_sol, Sigmadv_in, Sigmadv_out, Sigmapv_in, Sigmapv_out, Sigmagv_in, Sigmagv_out = \
                [], [], [], [], [Sigmap_in], [Sigmap_out], [Sigmag_in], [Sigmag_out]

     for i in range(len(t) - 1):

          dMgas = dMgas_dt(rin, rout, t[i], s, alpha, rhos, T0, betaT, mu, r1, C, Mstar, sigma) * (t[i + 1] - t[i])
          dMsol = dMsol_dt(rin, rout, t[i], s, alpha, Sigmap_in, Sigmap_out, rhos, T0, betaT, mu, r1, C, Mstar, sigma, vrw) * (t[i + 1] - t[i])

          dM_gas = np.append(dM_gas, dMgas)
          dM_sol = np.append(dM_sol, dMsol)

          #if constdtg == 0:
          Sigmap_in = Sigmap_in + dMsol / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
          Sigmap_out = Sigmap_out + dMsol / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
          
          Sigmag_in = Sigmag_in + dMgas / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
          Sigmag_out = Sigmag_out + dMgas / (np.pi * ((rout * cmperau)**2 - (rin * cmperau)**2))
          
          Sigmagv_in = np.append(Sigmagv_in, Sigmag_in)
          Sigmagv_out = np.append(Sigmagv_out, Sigmag_out)
              
          #else:
          #    Sigmap_in = Sigmadisk_act(rin, t[i], alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas
          #    Sigmap_out = Sigmadisk_act(rout, t[i], alpha, T0, betaT, mu, Mstar, r1, C) * dusttogas

          Sigmadv_in = np.append(Sigmadv_in, Sigmadisk_act(rin, t[i], alpha, T0, betaT, mu, Mstar, r1, C))
          Sigmadv_out = np.append(Sigmadv_out, Sigmadisk_act(rout, t[i], alpha, T0, betaT, mu, Mstar, r1, C))
          
          Sigmapv_in = np.append(Sigmapv_in, Sigmap_in)
          Sigmapv_out = np.append(Sigmapv_out, Sigmap_out)

     return dM_gas * f, dM_sol, Sigmadv_in, Sigmadv_out, Sigmapv_in, Sigmapv_out, Sigmagv_in, Sigmagv_out



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



     




        
        
