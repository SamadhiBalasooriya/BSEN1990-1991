from cmath import phase
import math
import numpy as np
import scipy
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.linalg import solve
from matrix import*
import matrix
from pedestrian import*
from sympy import *
from matplotlib import pyplot as plt


import pedestrian


def indicat(x, lb, numped):
    """"
    Indicator function which checks whether the pedestrian is or not on the bridge, it
    returns 1 if it is on the bridge, otherwise zero.
    Parameters
    ----------
    x : vector shows the positions of wheels
        Unit m.
    lb : single span length
        Unit m.
    numped : number of pedestrians
       
        Returns
    -------
    None
    """
    I = np.zeros((numped))  # important
    for i in range(numped):
        if 0 < x[i] < lb:
            I[i] = 1
        else:
            I[i] = 0
    return I

def Phi_x(x_interest, lb, rho, N_bridge):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x_interest : interest position at the bridge.
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
    # the end
    n = N_bridge
    phi_x = np.zeros((n, 1))
    for j in range(n):
        jt = j + 1
        phi_x[j] = (2 / rho / lb) ** 0.5 * sin(jt * pi * x_interest / lb)
    return phi_x

def Phi_x_data(x_interest, lb, modalmass, N_bridge,function_list):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x_interest : interest position at the bridge.
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
    # the end
    n = N_bridge
    phi_x = np.zeros((n, 1))
    for j in range(n):
        phi_x[j] = ((1/modalmass[j])**0.5)*function_list[j](x_interest)
    return phi_x



def Phi_matrix(xrb, lb, rho, N_bridge,numped):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x : position of pedestrians with time
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
   
    n = N_bridge
    N = np.zeros((n, numped))
    for i in range(n):
        for j in range(numped):
          it = i + 1
          N[i][j] = (2 / rho / lb) ** 0.5 * sin(it * pi * xrb[j] / lb) #x interest should be time dependednt x=vt xrb is a matrix with time varying location of each pedestrian
    I = indicat(xrb, lb, numped) 
    NN = np.dot(N, np.diag(I))
    return NN

def Phi_matrix_data(xrb, lb, modalmass, N_bridge, func_list):
    """
    Derive the mode shape of interested position at the bridge using different functions per row.

    Parameters
    ----------
    xrb : array-like
        Position of pedestrians with time.
        Unit m.
    lb : float
        Single span length.
        Unit m.
    rho : float
        modal mass of the bridge.
        Unit kg.
    N_bridge : int
        Number of modes taken into account.
    numped : int
        Number of pedestrians.
    func_list : list of functions
        List of functions to apply per row in the matrix.
        
    Returns
    -------
    NN : ndarray
        Mode shape matrix with dimensions (N_bridge, numped).
    """
    numped = len(xrb)
    n = N_bridge
    if len(func_list) != n:
        raise ValueError("The length of func_list must be equal to N_bridge.")
    
    N = np.zeros((n, numped))
    
    for i in range(n):
        for j in range(numped):
            # Apply the corresponding function from func_list[i]
            N[i][j] = ((1/modalmass[i])**0.5)*func_list[i](xrb[j])
            
    I = indicat(xrb, lb, numped)
    NN = np.dot(N, np.diag(I))
    
    return NN

def MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,t):
    """assembles the matrices with coupled SMD properties. Coupling is in M here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)
    NN = Phi_matrix(xrb, lb, rho, N_bridge,numped)
    v=Pedestrian.detVelocity
    
    #M assemble
    M3 =np.diag(mped)
    M1= np.hstack((mass,NN*M3)) 
    M2= np.zeros((numped,N_bridge))#(diag(mped).shape)
    M4 = np.hstack((M2,M3))
    M = np.vstack((M1,M4))
    #end

    #K assemble
    K2= np.zeros((N_bridge,numped))#(diag(k).shape)
    K3= np.hstack((k,K2))
    K4 = np.diag(kped)
    K5=np.hstack((-np.array(NN*K4).T,K4))
    K=np.vstack((K3,K5))
    #end

    #C assemble
    C2= np.zeros((N_bridge,numped))
    C3= np.hstack((c,C2))
    C4 = np.diag(cped)
    C5=np.hstack((-np.array(NN*C4).T,C4))
    C=np.vstack((C3,C5))
    #end
    
    #F matrix
    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
 
    
    #end
    return M,K,C,Ft


def Newmarksuper_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h



    Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)

    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)
    #print(Fc)
    Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
        
        Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
       
        Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
      
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu
    

def accdyn_super(bridge_instance,ddu, x_inter, hht):
    """
    Generate the acceleration vector (time) of the interested point
    at the bridge.
    Parameters
    ----------
    ddu : acceleration of the HSI system.
        Unit m/s^2.
    
    x_inter : dynamic of interest point at the bridge.
        Unit m.
    N_span : number of spans in the bridge.
        1 means single span
    v : pedestrian speed
        m/s.
    hht : time steps
        Unit s.
    -------
    None.
    """
    lb = bridge_instance.L
    rho = bridge_instance.rho
    N_bridge = bridge_instance.n
    v=Pedestrian.detVelocity

    phi_x = Phi_x(x_inter, lb, rho, N_bridge)
   
   
    meta = (np.diag(phi_x.flatten()).dot(ddu[:N_bridge, :]))
    

    column_sums = np.sum(meta, axis=0) 

    return column_sums





'''for pseudo excitation method'''

def Newmarkpseudo_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h


    Mc, Kc, Cc, _ = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)

    
    Fc=np.vstack((NN*0, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
       
        Mc, Kc, Cc,_= MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*force[:,[i]], np.zeros((numped, 1))))
       
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu

def Newmarkpseudo_HSI2(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h


    Mc, Kc, Cc, _ = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)

    
    Fc=np.vstack((NN*0, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
       
        Mc, Kc, Cc,_= MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*force[:,[i]], np.zeros((numped, 1))))
       
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu

def MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,t):
    """assembles the matrices with coupled SMD properties. Coupling is in K and C here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)
    NN = Phi_matrix(xrb, lb, rho, N_bridge,numped)
    v=Pedestrian.detVelocity
    kped = np.array(kped)
    cped = np.array(cped)
    #mass matrix
    M1 =np.diag(mped)
    M2= np.zeros((N_bridge,numped))
    M3= np.hstack((mass,M2)) 
    M4= np.hstack((M2.T,M1))
    M = np.vstack((M3,M4))
    #end

    #stiffness matrix
    
    K1 = np.diag(kped)
    K2 = -kped*NN
    K3 = -(NN.T)*kped[:,np.newaxis]   #-K1*(NN.T)
    K4= np.hstack((K3,K1))
    #for i in range(numped):
       # k += kped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    # Vectorized approach
    k += (NN * kped).dot(NN.T)
    K5 = np.hstack((k,K2))
    K = np.vstack((K5,K4))
    #end

    #C assemble
    C1 = np.diag(cped)
    C2 = -cped*NN
    C3 = -(NN.T)*cped[:,np.newaxis]
    C4= np.hstack((C3,C1))
    #for i in range(numped):
        #c += cped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    c += (NN * cped).dot(NN.T)
    C5 = np.hstack((c,C2))
    C = np.vstack((C5,C4))
    #end

    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    Ft=Ft*NN
    Fsum= np.sum(Ft,axis=1)
    F = Fsum[:, np.newaxis]
  
    return M,K,C,F

def MatrixAssemblesymetric_social(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, modalmass, N_bridge,numped,t,func_list):
    """assembles the matrices with coupled SMD properties. Coupling is in K and C here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix2(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)

    
    kped = np.array(kped)
    cped = np.array(cped)

    # -------- zero-pedestrian baseline --------
    if numped == 0:
        F = np.zeros((N_bridge, 1))
        return mass, k, c, F

    NN = Phi_matrix_data(xrb, lb, modalmass, N_bridge, func_list)

    #mass matrix
    M1 =np.diag(mped)
    M2= np.zeros((N_bridge,numped))
    M3= np.hstack((mass,M2)) 
    M4= np.hstack((M2.T,M1))
    M = np.vstack((M3,M4))
    #end

    #stiffness matrix
    
    K1 = np.diag(kped)
    K2 = -kped*NN
    K3 = -(NN.T)*kped[:,np.newaxis]   #-K1*(NN.T)
    K4= np.hstack((K3,K1))
    #for i in range(numped):
       # k += kped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    # Vectorized approach
    k += (NN * kped).dot(NN.T)
    K5 = np.hstack((k,K2))
    K = np.vstack((K5,K4))
    #end

    #C assemble
    C1 = np.diag(cped)
    C2 = -cped*NN
    C3 = -(NN.T)*cped[:,np.newaxis]
    C4= np.hstack((C3,C1))
    #for i in range(numped):
        #c += cped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    c += (NN * cped).dot(NN.T)
    C5 = np.hstack((c,C2))
    C = np.vstack((C5,C4))
    #end

    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    Ft=Ft*NN
    Fsum= np.sum(Ft,axis=1)
    F = Fsum[:, np.newaxis]
  
    return M,K,C,F

def MatrixAssemblesymetric_socialeeklo(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, modalmass, N_bridge,numped,t,func_list):
    """assembles the matrices with coupled SMD properties. Coupling is in K and C here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix3(self=case_bridge)
    c=bridge.Damp_matrix4(self=case_bridge)

    
    kped = np.array(kped)
    cped = np.array(cped)

    # -------- zero-pedestrian baseline --------
    if numped == 0:
        F = np.zeros((N_bridge, 1))
        return mass, k, c, F

    NN = Phi_matrix_data(xrb, lb, modalmass, N_bridge, func_list)

    #mass matrix
    M1 =np.diag(mped)
    M2= np.zeros((N_bridge,numped))
    M3= np.hstack((mass,M2)) 
    M4= np.hstack((M2.T,M1))
    M = np.vstack((M3,M4))
    #end

    #stiffness matrix
    
    K1 = np.diag(kped)
    K2 = -kped*NN
    K3 = -(NN.T)*kped[:,np.newaxis]   #-K1*(NN.T)
    K4= np.hstack((K3,K1))
    #for i in range(numped):
       # k += kped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    # Vectorized approach
    k += (NN * kped).dot(NN.T)
    K5 = np.hstack((k,K2))
    K = np.vstack((K5,K4))
    #end

    #C assemble
    C1 = np.diag(cped)
    C2 = -cped*NN
    C3 = -(NN.T)*cped[:,np.newaxis]
    C4= np.hstack((C3,C1))
    #for i in range(numped):
        #c += cped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    c += (NN * cped).dot(NN.T)
    C5 = np.hstack((c,C2))
    C = np.vstack((C5,C4))
    #end

    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    Ft=Ft*NN
    Fsum= np.sum(Ft,axis=1)
    F = Fsum[:, np.newaxis]
  
    return M,K,C,F

def udl_modal_participation(lb, modalmass, func_list, n_int=2001):
    """
    Compute Γ_j = ∫_0^L φhat_j(x) dx for each mode j,
    where φhat_j(x) = φ_j(x) / sqrt(M_j).

    Returns
    -------
    Gamma : ndarray, shape (N_bridge,)
    """
    x = np.linspace(0.0, lb, n_int)
    N_bridge = len(func_list)
    Gamma = np.zeros(N_bridge)

    for j in range(N_bridge):
        phi = func_list[j](x)                 # φ_j(x)
        phihat = phi / np.sqrt(modalmass[j])  # φhat_j(x)
        Gamma[j] = np.trapz(phihat, x)        # numerical integral

    return Gamma

def udl_modal_participation2(lb, modalmass, func_list, n_int=2001):
    """
    Code-style modal participation:
        Γ_j = ∫ |φhat_j(x)| dx

    Load sign follows the mode shape.
    """
    x = np.linspace(0.0, lb, n_int)
    N_bridge = len(func_list)
    Gamma = np.zeros(N_bridge)

    for j in range(N_bridge):
        phi = func_list[j](x)
        phihat = phi / np.sqrt(modalmass[j])

        # IMPORTANT CHANGE: absolute value
        Gamma[j] = np.trapz(np.abs(phihat), x)

    return Gamma


def center_abs_normalize(signal):
    signal = np.asarray(signal, dtype=float)

    midpoint = 0.5 * (np.max(signal) + np.min(signal))
    centered = signal - midpoint
    abs_centered = np.abs(centered)
    amplitude = np.max(abs_centered)

    if amplitude > 0:
        normalized = centered / amplitude
        abs_normalized = abs_centered / amplitude
    else:
        normalized = centered.copy()
        abs_normalized = abs_centered.copy()

    return midpoint, amplitude, centered, abs_centered, normalized, abs_normalized


def normalize_abs_zero_midpoint(signals, use_median=True):
    """
    signals: array of shape (n_signals, nt)

    Returns
    -------
    mid_axis      : shape (n_signals,)
    amplitude     : shape (n_signals,)
    centered      : shape (n_signals, nt)
    abs_centered  : shape (n_signals, nt)
    abs_normalized: shape (n_signals, nt)
    """
    signals = np.asarray(signals, dtype=float)

    if use_median:
        mid_axis = np.median(signals, axis=1, keepdims=True)
    else:
        mid_axis = 0.5 * (
            np.max(signals, axis=1, keepdims=True) +
            np.min(signals, axis=1, keepdims=True)
        )

    centered = signals - mid_axis
    abs_centered = np.abs(centered)

    amplitude = np.max(abs_centered, axis=1, keepdims=True)
    amplitude_safe = np.where(amplitude == 0, 1.0, amplitude)

    abs_normalized = abs_centered / amplitude_safe

    return (
        mid_axis[:, 0],
        amplitude[:, 0],
        centered,
        abs_centered,
        abs_normalized
    )

def modal_force_from_udl_2harm(t, width, xfreq, density, xi, n, S, lb, modalmass, func_list, phase=(0.0, 0.0), force_reduction_factor=1.0):
    """
    Treat force_2harm(...) as UDL q(t) [N/m] and compute modal force vector:

        F_j(t) = q(t) * Γ_j

    Returns
    -------
    F : ndarray, shape (N_bridge, 1)
    """
    q_t = force_2harm(t, xfreq, density, xi, S, phase=phase, rf=force_reduction_factor)  # scalar (UDL intensity)
    Gamma = udl_modal_participation2(lb, modalmass, func_list)    # (N_bridge,)
    F = (q_t* width * Gamma)[:, None]                                  # (N_bridge,1)
    return F

def MatrixAssembleCode(case_bridge, lb, modalmass, func_list,
                     t,width, xfreq, density, xi, n, S, phase=(0.0, 0.0),force_reduction_factor=1.0):
    """
    Assemble M, K, C and modal force vector F using UDL assumption.
    """
    M = bridge.Mass_matrix(self=case_bridge)
    K = bridge.Stiffness_matrix2(self=case_bridge)
    C = bridge.Damp_matrix2(self=case_bridge)

    F = modal_force_from_udl_2harm(t,width, xfreq, density, xi, n, S, lb, modalmass, func_list, phase=phase, force_reduction_factor=force_reduction_factor)
    return M, K, C, F



def MatrixAssembleMF(case_pedestrian,case_bridge,xrb, lb, modalmass, N_bridge,func_list,t): 
    """assembles the matrices with coupled SMD properties. Coupling is in K and C here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix2(self=case_bridge)
    c=bridge.Damp_matrix2(self=case_bridge)
    NN = Phi_matrix_data(xrb, lb, modalmass, N_bridge,func_list)
    

    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    Ft=Ft*NN
    Fsum= np.sum(Ft,axis=1)
    F = Fsum[:, np.newaxis]
  
    return mass,k,c,F
  
def Newmarksuper_HSI2(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """
    #absolute_max = max(xrb, key=abs)
    t = np.transpose(np.arange(0, (lb + 5) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

   

    Mc, Kc, Cc, Fc = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
   
    Fc=np.vstack((Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
    
        Mc, Kc, Cc, Fc = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
      
        Fc=np.vstack((Fc, np.zeros((numped, 1))))
        
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu

def Newmarksuper_HSIsocialeeklo(case_pedestrian, case_bridge, numped, N_bridge, lb, hht, v, mped, kped, cped, xrb_over_time, modalmass, func_list, simulation_duration):
    """
    Solve the coupled HSI matrices by Newmark-beta method, using precomputed pedestrian positions.

    Parameters
    ----------
    case_pedestrian : Pedestrian object
    case_bridge : Bridge object
    numped : int
        Number of pedestrians.
    N_bridge : int
        Number of bridge modes.
    lb : float
        Bridge length.
    hht : float
        Time step.
    v : float
        Velocity of pedestrians.
    mped, kped, cped : np.ndarray
        Mass, stiffness, and damping matrices for pedestrians.
    xrb_over_time : np.ndarray
        Array of pedestrian positions over time. Shape: (numped, time_steps).
    modalmass : float
        modalmass of bridge.

    Returns
    -------
    u, du, ddu : np.ndarray
        Displacement, velocity, and acceleration response of the system.
    """

    # Generate time array
    #simulation_duration = 700.0  # seconds

    t = np.arange(
        0.0,
        simulation_duration,
        hht
    )
    #t = np.transpose(np.arange(0, (lb + 88) / v, hht))  # Time duration of the simulation
    n = np.size(t)  # Number of time steps
    h = hht  # Time step size

    # Check input consistency
    #if xrb_over_time.shape[1] != n+1:
        #raise ValueError(f"xrb_over_time has shape {xrb_over_time.shape}, but expected (numped, {n+1})")

    # Newmark-beta parameters
    gamma = 1 / 2
    beta = 1 / 4

    # Newmark integration coefficients
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

    # Initial conditions
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))

    # Assemble initial system matrices
    Mc, Kc, Cc, Fc = MatrixAssemblesymetric_socialeeklo(case_pedestrian, case_bridge, mped, kped, cped, xrb_over_time[0, :], lb, modalmass, N_bridge, numped, 0, func_list)
    
    # Ensure force vector has correct shape
    Fc = np.vstack((Fc, np.zeros((numped, 1))))

    # Compute initial acceleration
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))

    # Initialize displacement, velocity, and acceleration arrays
    u = np.zeros((numped + N_bridge, n))
    du = np.zeros((numped + N_bridge, n))
    ddu = np.zeros((numped + N_bridge, n))

    # Set initial conditions
    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0

    # Time-stepping loop
    for i in range(n - 2):
        it = i + 1

        # Update pedestrian positions using precomputed values
        xr = xrb_over_time[it, :]  

        # Recalculate matrices with updated positions
        Mc, Kc, Cc, Fc = MatrixAssemblesymetric_socialeeklo(case_pedestrian, case_bridge, mped, kped, cped, xr, lb, modalmass, N_bridge, numped, t[i], func_list)
        Fc = np.vstack((Fc, np.zeros((numped, 1))))  # Ensure correct shape

        # Compute effective force
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )

        # Compute effective stiffness
        Keff = Kc + a0 * Mc + a1 * Cc
       
        # Solve for displacement
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        #u[:, [it]] = solve(Keff, Feff, assume_a="pos")

        # Update acceleration
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )

        # Update velocity
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    return u, du, ddu



def Newmarksuper_HSIsocial(case_pedestrian, case_bridge, numped, N_bridge, lb, hht, v, mped, kped, cped, xrb_over_time, modalmass, func_list):
    """
    Solve the coupled HSI matrices by Newmark-beta method, using precomputed pedestrian positions.

    Parameters
    ----------
    case_pedestrian : Pedestrian object
    case_bridge : Bridge object
    numped : int
        Number of pedestrians.
    N_bridge : int
        Number of bridge modes.
    lb : float
        Bridge length.
    hht : float
        Time step.
    v : float
        Velocity of pedestrians.
    mped, kped, cped : np.ndarray
        Mass, stiffness, and damping matrices for pedestrians.
    xrb_over_time : np.ndarray
        Array of pedestrian positions over time. Shape: (numped, time_steps).
    modalmass : float
        modalmass of bridge.

    Returns
    -------
    u, du, ddu : np.ndarray
        Displacement, velocity, and acceleration response of the system.
    """

    # Generate time array
    t = np.transpose(np.arange(0, (lb + 88) / v, hht))  # Time duration of the simulation
    n = np.size(t)  # Number of time steps
    h = hht  # Time step size

    # Check input consistency
    #if xrb_over_time.shape[1] != n+1:
        #raise ValueError(f"xrb_over_time has shape {xrb_over_time.shape}, but expected (numped, {n+1})")

    # Newmark-beta parameters
    gamma = 1 / 2
    beta = 1 / 4

    # Newmark integration coefficients
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

    # Initial conditions
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))

    # Assemble initial system matrices
    Mc, Kc, Cc, Fc = MatrixAssemblesymetric_social(case_pedestrian, case_bridge, mped, kped, cped, xrb_over_time[0, :], lb, modalmass, N_bridge, numped, 0, func_list)
    
    # Ensure force vector has correct shape
    Fc = np.vstack((Fc, np.zeros((numped, 1))))

    # Compute initial acceleration
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))

    # Initialize displacement, velocity, and acceleration arrays
    u = np.zeros((numped + N_bridge, n))
    du = np.zeros((numped + N_bridge, n))
    ddu = np.zeros((numped + N_bridge, n))

    # Set initial conditions
    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0

    # Time-stepping loop
    for i in range(n - 2):
        it = i + 1

        # Update pedestrian positions using precomputed values
        xr = xrb_over_time[it, :]  

        # Recalculate matrices with updated positions
        Mc, Kc, Cc, Fc = MatrixAssemblesymetric_social(case_pedestrian, case_bridge, mped, kped, cped, xr, lb, modalmass, N_bridge, numped, t[i], func_list)
        Fc = np.vstack((Fc, np.zeros((numped, 1))))  # Ensure correct shape

        # Compute effective force
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )

        # Compute effective stiffness
        Keff = Kc + a0 * Mc + a1 * Cc
       
        # Solve for displacement
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        #u[:, [it]] = solve(Keff, Feff, assume_a="pos")

        # Update acceleration
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )

        # Update velocity
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    return u, du, ddu


def Newmarksuper_Code(
    case_bridge,
    N_bridge,
    lb,
    hht,
    t_end,
    width,
    xfreq,
    density,
    xi,
    n_peds,
    S,
    modalmass,
    func_list,
    phase=(0.0, 0.0),
    t_start=0.0,
    force_reduction_factor=1.0
):
    """
    Newmark-beta integration for bridge modal DOFs with a 2-harmonic UDL load.
    No pedestrian positions are needed.

    Parameters
    ----------
    case_bridge : Bridge object
    N_bridge : int
        Number of modes.
    lb : float
        Span length [m] (used in modal participation integral)
    hht : float
        Time step [s]
    t_end : float
        End time of simulation [s]
    width : float
        Effective loaded width [m] to convert N/m^2 -> N/m
    xfreq : float
        First harmonic frequency (Hz) (often fs); second is 2*xfreq
    density : float
        Pedestrian density [ped/m^2]
    xi : float or array-like
        Modal damping ratio(s). If array-like length N_bridge, you can handle per-mode later.
        For now: use scalar as per your current UDL choice.
    n_peds : float
        Total number of pedestrians n [-]
    S : float
        Deck area [m^2]
    modalmass : array-like
        Modal masses (length N_bridge)
    func_list : list[callable]
        Mode shape functions φ_j(x)
    phase : tuple
        (phi1, phi2) phase angles [rad]
    t_start : float
        Start time [s] (default 0)

    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time vector
    u, du, ddu : ndarrays, shape (N_bridge, n_steps)
        Modal displacement, velocity, acceleration
    """

    # ---- time vector for the selected window ----
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start.")

    t = np.arange(t_start, t_end + hht, hht)
    n_steps = t.size
    h = hht

    # ---- Newmark-beta params (average acceleration) ----
    gamma = 1 / 2
    beta = 1 / 4

    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

    # ---- initial conditions ----
    u = np.zeros((N_bridge, n_steps))
    du = np.zeros((N_bridge, n_steps))
    ddu = np.zeros((N_bridge, n_steps))

    u0 = np.zeros((N_bridge, 1))
    du0 = np.zeros((N_bridge, 1))

    # ---- assemble constant structural matrices once ----
    M = bridge.Mass_matrix(self=case_bridge)
    K = bridge.Stiffness_matrix3(self=case_bridge)
    C = bridge.Damp_matrix3(self=case_bridge)

    
    # ---- initial acceleration ----
    F0 = modal_force_from_udl_2harm(t[0], width, xfreq, density, xi, n_peds, S, lb, modalmass, func_list, phase=phase, force_reduction_factor=force_reduction_factor)  # load at t[0]
    ddu0 = np.linalg.solve(M, F0 - C @ du0 - K @ u0)

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0

    # ---- time stepping ----
    for i in range(n_steps - 1):
        it = i + 1

        Fc = modal_force_from_udl_2harm(t[i], width, xfreq, density, xi, n_peds, S, lb, modalmass, func_list, phase=phase, force_reduction_factor=force_reduction_factor)  # load at current time (you could also use t[it])

        Feff = (
            Fc
            + M @ (a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + C @ (a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )

        Keff = K + a0 * M + a1 * C

        u[:, [it]] = np.linalg.solve(Keff, Feff)

        ddu[:, [it]] = a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    return t, u, du, ddu

def Newmarksuper_MF(case_pedestrian, case_bridge, numped, N_bridge, lb, hht, v, mped, xrb_over_time, modalmass, func_list):
    """
    Solve the coupled HSI matrices by Newmark-beta method, using precomputed pedestrian positions.

    Parameters
    ----------
    case_pedestrian : Pedestrian object
    case_bridge : Bridge object
    numped : int
        Number of pedestrians.
    N_bridge : int
        Number of bridge modes.
    lb : float
        Bridge length.
    hht : float
        Time step.
    v : float
        Velocity of pedestrians.
    mped, kped, cped : np.ndarray
        Mass, stiffness, and damping matrices for pedestrians.
    xrb_over_time : np.ndarray
        Array of pedestrian positions over time. Shape: (numped, time_steps).
    modalmass : float
        modalmass of bridge.

    Returns
    -------
    u, du, ddu : np.ndarray
        Displacement, velocity, and acceleration response of the system.
    """

    # Generate time array
    t = np.transpose(np.arange(0, (lb + 5) / v, hht))  # Time duration of the simulation
    n = np.size(t)  # Number of time steps
    h = hht  # Time step size

    # Check input consistency
    #if xrb_over_time.shape[1] != n+1:
        #raise ValueError(f"xrb_over_time has shape {xrb_over_time.shape}, but expected (numped, {n+1})")

    # Newmark-beta parameters
    gamma = 1 / 2
    beta = 1 / 4

    # Newmark integration coefficients
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

    # Initial conditions
    u0 = np.zeros((N_bridge, 1))
    du0 = np.zeros((N_bridge, 1))

    # Assemble initial system matrices
    Mc, Kc, Cc, Fc = MatrixAssembleMF(case_pedestrian, case_bridge, xrb_over_time[0, :], lb, modalmass, N_bridge, func_list,t[0])
    
    # Ensure force vector has correct shape
    #Fc = np.vstack((Fc, np.zeros((numped, 1))))

    # Compute initial acceleration
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))

    # Initialize displacement, velocity, and acceleration arrays
    u = np.zeros((N_bridge, n))
    du = np.zeros((N_bridge, n))
    ddu = np.zeros((N_bridge, n))

    # Set initial conditions
    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0

    # Time-stepping loop
    for i in range(n - 2):
        it = i + 1

        # Update pedestrian positions using precomputed values
        xr = xrb_over_time[it, :]  

        # Recalculate matrices with updated positions 
        Mc, Kc, Cc, Fc = MatrixAssembleMF(case_pedestrian, case_bridge, xr, lb, modalmass, N_bridge, func_list, t[i] )
        #Fc = np.vstack((Fc, np.zeros((numped, 1))))  # Ensure correct shape

        # Compute effective force
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )

        # Compute effective stiffness
        Keff = Kc + a0 * Mc + a1 * Cc
       
        # Solve for displacement
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        #u[:, [it]] = solve(Keff, Feff, assume_a="pos")

        # Update acceleration
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )

        # Update velocity
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    return u, du, ddu

def accdyn_super_social(bridge_instance,ddu, x_inter, modalmass,func_list):
    """
    Generate the acceleration vector (time) of the interested point
    at the bridge.
    Parameters
    ----------
    ddu : acceleration of the HSI system.
        Unit m/s^2.
    
    x_inter : dynamic of interest point at the bridge.
        Unit m.
    N_span : number of spans in the bridge.
        1 means single span
    v : pedestrian speed
        m/s.
    hht : time steps
        Unit s.
    -------
    None.
    """
    lb = bridge_instance.L
    rho = bridge_instance.rho
    N_bridge = bridge_instance.n
    v=Pedestrian.detVelocity

    phi_x = Phi_x_data(x_inter, lb, modalmass, N_bridge,func_list)
   
   
    meta = (np.diag(phi_x.flatten()).dot(ddu[:N_bridge, :]))
    

    column_sums = np.sum(meta, axis=0) 

    return column_sums


def compute_1sec_rms_mean(results, sampling_rate):
    window_size = sampling_rate
    num_sims, num_steps = results.shape
    num_windows = num_steps // window_size

    all_rms_means = []

    for sim in range(num_sims):
        sim_rms_values = []
        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            window_data = results[sim, start:end]
            rms = np.sqrt(np.mean(window_data ** 2))
            sim_rms_values.append(rms)
        mean_rms = np.mean(sim_rms_values)
        all_rms_means.append(mean_rms)

    return np.mean(all_rms_means)

def calc_frf(M, C, K, freq_range,):
    """
    Calculate the |Frequency Response Function (FRF)|**2 for a MDOF system.
    
    Parameters:
    M :Mass matrix.

    C :Damping matrix.

    K :Stiffness matrix.

    freq_range : ndarray Array of frequencies at which to calculate the FRF.(rad/s note:that not in Hz)
    
    t= timepoint being considerd 
    
    Returns:
    FRF : ndarray
        Frequency Response Function matrix, each column corresponds to a frequency.
    """
    n_dof = M.shape[0]
    FRF = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    accelerance = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    
    for i, omega in enumerate(freq_range):
        omega_squared = omega**2
        H = K - omega_squared * M + 1j * omega * C  # Complex dynamic stiffness matrix
        FRF[:, :, i] = np.linalg.inv(H)
        accelerance [:, :, i] = -omega_squared * FRF[:, :, i]

    return FRF, accelerance
    
    


def g_pj(omega_j, N, L, vm):

    """
    Calculate the g_pj value based on the given parameters.

    This function computes the g_pj value, which is typically used in the analysis of
    structural responses. The g_pj value is calculated using the frequency `fe`, 
    the period `T`, and the given angular frequency `ωj`.

    Parameters:
    -----------
    omega_j : float
        Angular frequency (ωj) in radians per second.
    N : int or float
        Number of cycles.
    L : float
        Length in meters.
    vm : float
        Velocity in meters per second.

    Returns:
    --------
    g_pj_value : float
        The computed g_pj value based on the provided input parameters.
    """

    # Calculate fe
    fe = omega_j / (2 * np.pi)
    
    # Calculate T
    T = (N * L) / vm
    
    # Calculate g_pj
    g_pj_value = (np.sqrt(2 * np.log(2* fe * T)) + 0.5772 / np.sqrt(2 * np.log(2 * fe * T))) 
    
    return g_pj_value

def calculate_frf_and_accelerance(M, C, K, frequencies):
    """
    Calculate the Frequency Response Function (FRF) and accelerance for given M, C, K matrices over a range of frequencies.

    Parameters:
    M (numpy.ndarray): Mass matrix
    C (numpy.ndarray): Damping matrix
    K (numpy.ndarray): Stiffness matrix
    frequencies (numpy.ndarray): Array of frequencies

    Returns:
    tuple: (FRF, accelerance) where both are numpy.ndarrays of shape (len(frequencies), M.shape[0], M.shape[1])
    """
    # Initialize arrays to store the FRF and accelerance values
    FRF = np.zeros((len(frequencies), M.shape[0], M.shape[1]), dtype=complex)
    accelerance = np.zeros((len(frequencies), M.shape[0], M.shape[1]), dtype=complex)

    # Loop over each frequency to calculate the FRF and accelerance
    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq  # Angular frequency
        H = np.linalg.inv(K - omega**2 * M + 1j * omega * C)  # Calculate the FRF
        FRF[i, :, :] = H
        accelerance[i, :, :] = -omega**2 * H

    return FRF, accelerance

def calculate_response_std(M, C, K, frequencies, input_psd,n_dof):
    """
    Calculate the response_std given the input PSD and the M, C, K matrices over a range of frequencies.

    Parameters:
    M (numpy.ndarray): Mass matrix
    C (numpy.ndarray): Damping matrix
    K (numpy.ndarray): Stiffness matrix
    frequencies (numpy.ndarray): Array of frequencies
    input_psd (numpy.ndarray): Input PSD array of shape (len(frequencies),)

    Returns:
    numpy.ndarray: Response PSD array of shape (len(frequencies), M.shape[0], M.shape[1])
    """
    # Calculate the FRF and accelerance
    _, accelerance = calc_frf(M, C, K, frequencies)

    # Calculate the magnitude squared of the FRF
    accelerance_magnitude_squared = np.abs(accelerance)**2
    #FRF_magnitude_squared = np.abs(FRF)**2

    # Calculate the response_std
    deltaF= frequencies[4]-frequencies[3]
    #n_dof = M.shape[0]
    for i in range(n_dof):
        multiply = accelerance_magnitude_squared[[i],[i],:].flatten() * input_psd[[i],:]
    
    sigma2=np.trapz(multiply, dx=deltaF, axis=0)
    
    
    return np.sqrt(sigma2)

def plot_frf_magnitude(M, C, K, freq_range):
    """
    Plot the FRF magnitude of a multi-degree-of-freedom system.
    
    Parameters:
    M : ndarray
        Mass matrix.
    C : ndarray
        Damping matrix.
    K : ndarray
        Stiffness matrix.
    freq_range : ndarray
        Array of frequencies at which to calculate the FRF.
    
    Returns:
    None. The function plots the FRF magnitude for each degree of freedom.
    """
    n_dof = M.shape[0]
    FRF = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    
    for i, omega in enumerate(freq_range):
        omega_squared = omega**2
        H = K - omega_squared * M + 1j * omega * C  # Complex dynamic stiffness matrix
        FRF[:, :, i] = np.linalg.inv(H)
    
    # Plot the magnitude of the FRF for each degree of freedom
    #plt.figure(figsize=(10, 6))
    for dof in range(n_dof):
        plt.plot(freq_range, np.abs(FRF[dof, dof, :]), label=f'DOF {dof + 1}')
    
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('FRF Magnitude')
    plt.title('Frequency Response Function (FRF) Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def montecarlo_stocastic_accn(length, modulus, linearMass, modalDampingRatio, numbers, pedmass, peddamp, pedBodyF, pedvelocity, numped, hht, x_interested,i):
    import random
    acceleration_responses = np.zeros((100, len(x_interested)))
    mean_pace = 2 #Hz  2005 pachi
    pace_COV = 0.01

    mean_mass= 70 #kg
    mass_COV= 0.17 #from butz 2008

    mean_velocity = 1.3
    std_velocity = 0.12 #pachi 2005

    # Randomize pedestrian parameters: mass, and pace
    randomPace = random.gauss(mean_pace, pace_COV*mean_pace)
    randomMass = random.gauss(mean_mass, mass_COV*mean_mass) # Damping ratio (mean=0.3, std=0.05)
    
    # Calculate pedestrian stiffness and damping based on random parameters
    kped = (2 * np.pi * pedBodyF) ** 2 * mean_mass
    cped = (2 * np.pi * pedBodyF) * 2 * peddamp * pedmass

    # Convert to arrays for compatibility with the solver
    mped = np.array([pedmass])
    cped = np.array([cped])
    kped = np.array([kped])
    xrb = [0]  # Initial position

    # Create bridge and pedestrian instances
    Bridge = bridge(
            length=length,
            modulus=modulus,
            density=linearMass,
            damp=modalDampingRatio,
            numbers=numbers
                        )

    Human = Pedestrian(
            mass=pedmass,
            damp=peddamp,
            stiff=kped,
            pace=randomPace,
            phase=0,
            location=0,
            velocity=pedvelocity,
            iSync=0
        )

    # Solve for acceleration response with Human-Structure Interaction (HSI)
    _, _, ddu_hsi = Newmarksuper_HSI(Human, Bridge, numped, numbers, length, hht, pedvelocity, mped, kped, cped, xrb, linearMass)
    accn_hsi = accdyn_super(Bridge, ddu_hsi, x_interested, hht)

    # Store the acceleration response for this simulation
    acceleration_responses[i, :] = accn_hsi

    return acceleration_responses

def assemble_state_space(M, C, K):
    """
    From M, C, K build A, B for:
        z' = A z + B f
    with z = [q; qdot]
    """
    n = M.shape[0]
    I = np.eye(n)
    Minv = np.linalg.inv(M)

    A = np.block([
        [np.zeros((n, n)), I],
        [-Minv @ K,        -Minv @ C]
    ])
    B = np.vstack([
        np.zeros((n, n)),
        Minv
    ])
    return A, B

def step_implicit_euler(A, B, z_k, f_k1, dt):
    """
    One implicit Euler step:
        z_{k+1} = (I - dt A)^{-1} (z_k + dt B f_{k+1})
    If f_k1 is None, treat as zero.
    """
    n = A.shape[0]
    I = np.eye(n)
    rhs = z_k.copy()
    if f_k1 is not None:
        rhs = rhs + dt * (B @ f_k1)

    z_k1 = np.linalg.solve(I - dt * A, rhs)
    return z_k1



def solve_HSI_state_space(
        Human, Bridge,
        mped, kped, cped,
        xrb, length,
        modalmass, numbers, numped,
        t, func_list,
        dt
    ):
    nt = len(t)

    # ---- 1) Do ONE assembly to detect system size ----
    M0, K0, C0, F0_modal = MatrixAssemblesymetric_social(
        Human, Bridge,
        mped, kped, cped,
        xrb[0, :], length,
        modalmass, numbers, numped,
        t[0], func_list
    )

    ndof = M0.shape[0]        # TRUE number of DOFs (bridge + humans etc.)
    z = np.zeros(2 * ndof)    # [q0; qdot0]
    z_hist = np.zeros((nt, 2 * ndof))
    qdd_hist = np.zeros((nt, ndof))

    for i in range(nt):
        ti = t[i]
        xi = xrb[i, :]

        M, K, C, F_modal = MatrixAssemblesymetric_social(
            Human, Bridge,
            mped, kped, cped,
            xi, length,
            modalmass, numbers, numped,
            ti, func_list
        )

        # --- hard consistency checks ---
        if M.shape[0] != ndof or M.shape[1] != ndof:
            raise RuntimeError(
                f"STEP {i}: M changed size. Initial ndof={ndof}, "
                f"M.shape={M.shape}"
            )
        if C.shape != M.shape or K.shape != M.shape:
            raise RuntimeError(
                f"STEP {i}: C or K shape mismatch. "
                f"M.shape={M.shape}, C.shape={C.shape}, K.shape={K.shape}"
            )
        # --- build full force vector F(ti) in all DOFs ---
        F_modal = F_modal.flatten()              # (N_bridge,)
        F = np.zeros(ndof)                       # (ndof,)
        F[:len(F_modal)] = F_modal              # [F_modal; 0_ped]

        A, B = assemble_state_space(M, C, K)
        

        z = step_implicit_euler(A, B, z, F, dt)
        z_hist[i, :] = z

        q  = z[:ndof]
        qd = z[ndof:]

        # extra debugging prints – TEMPORARY
        # reconstruct qdd from second-order equation with same F
        rhs = F - C @ qd - K @ q
        qdd = np.linalg.solve(M, rhs)
        qdd_hist[i, :] = qdd

    q_hist  = z_hist[:, :ndof]
    qd_hist = z_hist[:, ndof:]
    return q_hist, qd_hist, qdd_hist

def step_trapezoidal(A_prev, B_prev, F_prev,
                     A_curr, B_curr, F_curr,
                     x_k, dt):
    """
    One Crank–Nicolson / trapezoidal step for x' = A(t)x + B(t)F(t):

        (I - dt/2 A_{k+1}) x_{k+1}
          = (I + dt/2 A_k) x_k
            + dt/2 (B_k F_k + B_{k+1} F_{k+1})

    Parameters
    ----------
    A_prev, B_prev, F_prev : matrices/vectors at t_k
    A_curr, B_curr, F_curr : matrices/vectors at t_{k+1}
    x_k                    : state at t_k
    dt                     : time step

    Returns
    -------
    x_{k+1}
    """
    n = A_curr.shape[0]
    I = np.eye(n)

    rhs = (I + 0.5 * dt * A_prev) @ x_k \
          + 0.5 * dt * (B_prev @ F_prev + B_curr @ F_curr)

    x_k1 = np.linalg.solve(I - 0.5 * dt * A_curr, rhs)
    return x_k1

def solve_HSI_state_space2(
        Human, Bridge,
        mped, kped, cped,
        xrb, length,
        modalmass, numbers, numped,
        t, func_list,
        dt
    ):
    nt = len(t)

    # --- initial assembly ---
    M0, K0, C0, F0_modal = MatrixAssemblesymetric_social(
        Human, Bridge,
        mped, kped, cped,
        xrb[0, :], length,
        modalmass, numbers, numped,
        t[0], func_list
    )
    ndof = M0.shape[0]
    z = np.zeros(2 * ndof)          # [q0; qdot0]
    z_hist   = np.zeros((nt, 2 * ndof))
    qdd_hist = np.zeros((nt, ndof))

    # full force at step 0
    F0_modal = F0_modal.flatten()
    F0 = np.zeros(ndof)
    F0[:len(F0_modal)] = F0_modal

    A0, B0 = assemble_state_space(M0, C0, K0)

    # FIRST step: simple implicit Euler to get started
    z = step_implicit_euler(A0, B0, z, F0, dt)
    z_hist[0, :] = z
    q0  = z[:ndof]
    qd0 = z[ndof:]
    rhs0 = F0 - C0 @ qd0 - K0 @ q0
    qdd0 = np.linalg.solve(M0, rhs0)
    qdd_hist[0, :] = qdd0

    A_prev, B_prev, F_prev = A0, B0, F0

    # ---- time stepping, k = 1..nt-1 ----
    for k in range(1, nt):
        tk = t[k]
        xk = xrb[k, :]

        M, K, C, F_modal = MatrixAssemblesymetric_social(
            Human, Bridge,
            mped, kped, cped,
            xk, length,
            modalmass, numbers, numped,
            tk, func_list
        )

        ndof_now = M.shape[0]
        if ndof_now != ndof:
            raise RuntimeError(f"STEP {k}: M changed size: {ndof_now} vs {ndof}")

        # full force at this step
        F_modal = F_modal.flatten()
        F_curr = np.zeros(ndof)
        F_curr[:len(F_modal)] = F_modal

        A_curr, B_curr = assemble_state_space(M, C, K)

        # trapezoidal step
        z = step_trapezoidal(A_prev, B_prev, F_prev,
                             A_curr, B_curr, F_curr,
                             z, dt)

        z_hist[k, :] = z
        q  = z[:ndof]
        qd = z[ndof:]

        rhs = F_curr - C @ qd - K @ q
        qdd = np.linalg.solve(M, rhs)
        qdd_hist[k, :] = qdd

        # shift
        A_prev, B_prev, F_prev = A_curr, B_curr, F_curr

    q_hist  = z_hist[:, :ndof]
    qd_hist = z_hist[:, ndof:]
    return q_hist, qd_hist, qdd_hist

def solve_HSI_state_space_full(
        Human, Bridge,
        mped, kped, cped,
        xrb, length,
        modalmass, numbers, numped,
        t, func_list,
        dt,
        sigma_force=0.0,
        return_AB=False   # <--- NEW FLAG
    ):
    nt = len(t)

    # --- initial assembly ---
    M0, K0, C0, F0_modal = MatrixAssemblesymetric_social(
        Human, Bridge,
        mped, kped, cped,
        xrb[0, :], length,
        modalmass, numbers, numped,
        t[0], func_list
    )
    ndof = M0.shape[0]

    # state vector z = [q; qdot], size 2*ndof
    z = np.zeros(2 * ndof)
    z_hist   = np.zeros((nt, 2 * ndof))
    qdd_hist = np.zeros((nt, ndof))

    # optional storage for A,B
    A_hist = None
    B_hist = None
    if return_AB:
        # A: (2ndof, 2ndof), B: (2ndof, ndof)
        A_hist = np.zeros((nt, 2 * ndof, 2 * ndof))
        B_hist = np.zeros((nt, 2 * ndof, ndof))

    # full force at step 0
    F0_modal = F0_modal.flatten()
    F0 = np.zeros(ndof)
    F0[:len(F0_modal)] = F0_modal

    # --- NEW: add white noise to generalized force at step 0 ---
    if sigma_force > 0.0:
        noise0 = sigma_force * np.sqrt(dt) * np.random.randn(ndof)
        F0_eff = F0 + noise0
    else:
        F0_eff = F0
    A0, B0 = assemble_state_space(M0, C0, K0)

    if return_AB:
        A_hist[0, :, :] = A0
        B_hist[0, :, :] = B0

    # FIRST step: implicit Euler bootstrap
    z = step_implicit_euler(A0, B0, z, F0_eff, dt)
    z_hist[0, :] = z
    q0  = z[:ndof]
    qd0 = z[ndof:]
    rhs0 = F0_eff - C0 @ qd0 - K0 @ q0
    qdd0 = np.linalg.solve(M0, rhs0)
    qdd_hist[0, :] = qdd0

    A_prev, B_prev, F_prev = A0, B0, F0_eff

    # ---- time stepping, k = 1..nt-1 ----
    for k in range(1, nt):
        tk = t[k]
        xk = xrb[k, :]

        M, K, C, F_modal = MatrixAssemblesymetric_social(
            Human, Bridge,
            mped, kped, cped,
            xk, length,
            modalmass, numbers, numped,
            tk, func_list
        )

        ndof_now = M.shape[0]
        if ndof_now != ndof:
            raise RuntimeError(f"STEP {k}: M changed size: {ndof_now} vs {ndof}")

        # full force at this step
        F_modal = F_modal.flatten()
        F_curr = np.zeros(ndof)
        F_curr[:len(F_modal)] = F_modal

        # --- NEW: add white noise to generalized force at this step ---
        if sigma_force > 0.0:
            noise = sigma_force * np.sqrt(dt) * np.random.randn(ndof)
            F_eff = F_curr + noise
        else:
            F_eff = F_curr

        A_curr, B_curr = assemble_state_space(M, C, K)

        if return_AB:
            A_hist[k, :, :] = A_curr
            B_hist[k, :, :] = B_curr

        # trapezoidal (Crank–Nicolson) step in state space
        z = step_trapezoidal(
            A_prev, B_prev, F_prev,
            A_curr, B_curr, F_eff,
            z, dt
        )

        z_hist[k, :] = z
        q  = z[:ndof]
        qd = z[ndof:]

        rhs = F_eff - C @ qd - K @ q
        qdd = np.linalg.solve(M, rhs)
        qdd_hist[k, :] = qdd

        # shift
        A_prev, B_prev, F_prev = A_curr, B_curr, F_eff

    q_hist  = z_hist[:, :ndof]
    qd_hist = z_hist[:, ndof:]

    if return_AB:
        return q_hist, qd_hist, qdd_hist, A_hist, B_hist
    else:
        return q_hist, qd_hist, qdd_hist

def compute_1sec_rms_band(acc_all, t_ref):
    """
    acc_all: (N_sim, nt) array of acceleration time histories
    t_ref:   (nt,) time vector

    Returns:
        mean_rms_full, q05_rms_full, q95_rms_full  (all length nt)
    using a 1-second sliding window, padded at the start.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    dt = t_ref[1] - t_ref[0]
    samples_per_sec = int(round(1.0 / dt))

    # square of accelerations
    acc_sq = acc_all**2   # (N_sim, nt)

    # sliding windows of length 1 s along time axis
    windows = sliding_window_view(acc_sq, window_shape=samples_per_sec, axis=1)
    # RMS per sim, per time step (window ending at t_k)
    rms_all = np.sqrt(windows.mean(axis=-1))   # (N_sim, nt - samples_per_sec + 1)

    # stats across simulations
    mean_rms = rms_all.mean(axis=0)
    q05_rms  = np.quantile(rms_all, 0.05, axis=0)
    q95_rms  = np.quantile(rms_all, 0.95, axis=0)

    # pad to full time length for plotting
    nt   = t_ref.size
    idx0 = samples_per_sec - 1  # first index where RMS is defined

    mean_rms_full = np.empty(nt)
    q05_rms_full  = np.empty(nt)
    q95_rms_full  = np.empty(nt)

    # for t < 1 s, hold the first RMS value
    mean_rms_full[:idx0] = mean_rms[0]
    q05_rms_full[:idx0]  = q05_rms[0]
    q95_rms_full[:idx0]  = q95_rms[0]

    # from t >= 1 s, use true 1-sec RMS
    mean_rms_full[idx0:] = mean_rms
    q05_rms_full[idx0:]  = q05_rms
    q95_rms_full[idx0:]  = q95_rms

    return mean_rms_full, q05_rms_full, q95_rms_full




def _compute_stats(acc_all, t_ref):
    """Helper: mean / 5–95% cloud + 1-sec RMS band."""
    mean_acc = acc_all.mean(axis=0)
    q05_acc  = np.quantile(acc_all, 0.05, axis=0)
    q95_acc  = np.quantile(acc_all, 0.95, axis=0)

    mean_rms, q05_rms, q95_rms = compute_1sec_rms_band(acc_all, t_ref)
    return mean_acc, q05_acc, q95_acc, mean_rms, q05_rms, q95_rms


def plot_state_newmark(
    t_ref,
    acc_state_all=None,
    acc_newmark_all=None,
    which="both",         # "state", "newmark", "both"
    layout="combined"     # "combined" or "separate"
):
    """
    Plot time-history 5–95% cloud + 1-sec RMS for
    state-space and/or Newmark.

    which  : "state", "newmark", or "both"
    layout : "combined" (same axes) or "separate" (side-by-side subplots)
    """
    # REQUIRED INPUTS (already computed in your MC script):
    #   t_ref            : (nt,)   time vector
    #   acc_state_all    : (N_sim, nt) state-space accelerations
    #   acc_newmark_all  : (N_sim, nt) Newmark accelerations
    #
    # MAIN FUNCTION:
    #   plot_state_newmark(
    #       t_ref,
    #       acc_state_all=None,
    #       acc_newmark_all=None,
    #       which="both",        # "state", "newmark", "both"
    #       layout="combined"    # "combined" or "separate"
    #   )
    #
        # EXAMPLES:
    #   1) Both methods on same axes:
    #        plot_state_newmark(t_ref, acc_all, acc_newmark_all,
    #                           which="both", layout="combined")
    #
    #   2) Side-by-side: left = state-space, right = Newmark
    #        plot_state_newmark(t_ref, acc_all, acc_newmark_all,
    #                           which="both", layout="separate")
    #
    #   3) Only state-space:
    #        plot_state_newmark(t_ref, acc_state_all=acc_all,
    #                           which="state")
    #
    #   4) Only Newmark:
    #        plot_state_newmark(t_ref, acc_newmark_all=acc_newmark_all,
    #                           which="newmark")
    #
    # To customise:
    #   • Change colours / linestyles inside plot_method(...)
    #   • Uncomment the "mean acc" line if you also want the mean
    #     time history plotted over the cloud.
    # ============================================================
    show_state   = (which in ("state", "both")) and (acc_state_all is not None)
    show_newmark = (which in ("newmark", "both")) and (acc_newmark_all is not None)

    if not (show_state or show_newmark):
        raise ValueError("Nothing to plot: check `which` and provided arrays.")

    # --- compute stats ---
    if show_state:
        (mean_acc_s, q05_acc_s, q95_acc_s,
         mean_rms_s, q05_rms_s, q95_rms_s) = _compute_stats(acc_state_all, t_ref)

    if show_newmark:
        (mean_acc_n, q05_acc_n, q95_acc_n,
         mean_rms_n, q05_rms_n, q95_rms_n) = _compute_stats(acc_newmark_all, t_ref)

    # --- set up axes ---
    if layout == "separate" and show_state and show_newmark:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        axes = []
        if show_state:
            axes.append(("State-space", ax1))
        if show_newmark:
            axes.append(("Newmark", ax2))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        axes = []
        if show_state:
            axes.append(("State-space", ax))
        if show_newmark and not show_state:
            axes.append(("Newmark", ax))
        elif show_newmark and show_state and layout == "combined":
            axes.append(("Combined", ax))

    # --- plotting helper for one method ---
    def plot_method(ax, label_prefix,
                    mean_acc, q05_acc, q95_acc,
                    mean_rms, q05_rms, q95_rms,
                    style_mean, color_cloud, style_rms="solid"):
        # RMS
        ax.plot(t_ref, mean_rms, style_rms,
                label=f"Mean 1-sec RMS ({label_prefix})", linewidth=1.5)
        ax.fill_between(t_ref, q05_rms, q95_rms,
                        alpha=0.2, label=f"5–95% RMS band ({label_prefix})")
        # acceleration cloud
        ax.fill_between(t_ref, q05_acc, q95_acc,
                        color=color_cloud, alpha=0.3,
                        label=f"5–95% acc band ({label_prefix})")
        # mean acc if you want it:
        # ax.plot(t_ref, mean_acc, style_mean,
        #         label=f"Mean acc ({label_prefix})", linewidth=1.0)

    # --- draw ---
    for label, ax in axes:
        if label in ("State-space", "Combined") and show_state:
            plot_method(ax, "state-space",
                        mean_acc_s, q05_acc_s, q95_acc_s,
                        mean_rms_s, q05_rms_s, q95_rms_s,
                        style_mean="-", color_cloud="orange", style_rms="-")

        if label in ("Newmark", "Combined") and show_newmark:
            plot_method(ax, "Newmark",
                        mean_acc_n, q05_acc_n, q95_acc_n,
                        mean_rms_n, q05_rms_n, q95_rms_n,
                        style_mean="--", color_cloud="grey", style_rms="--")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mid-span acceleration / 1-sec RMS (m/s²)")
        ax.grid(True)
        ax.legend()
        if label == "Combined":
            ax.set_title("State-space vs Newmark")
        else:
            ax.set_title(label)

    fig.tight_layout()
    plt.show()


def ModalForceTimeHistory(case_pedestrian, case_bridge, mped, kped, cped,
                          xrb_over_time, lb, modalmass, N_bridge, numped,
                          t_array, func_list):
    """
    Generate the bridge/modal force time history.

    Parameters
    ----------
    case_pedestrian : Pedestrian object
    case_bridge : Bridge object
    mped, kped, cped : arrays
        Pedestrian mass, stiffness, damping properties.
    xrb_over_time : ndarray
        Pedestrian positions over time.
        Shape:
            - (numped, nt) if pedestrian positions vary with time
            - (numped,)    if positions are fixed
    lb : float
        Bridge length.
    modalmass : float or array
        Modal mass input used in Phi_matrix_data.
    N_bridge : int
        Number of bridge modes.
    numped : int
        Number of pedestrians.
    t_array : ndarray
        Time vector of length nt.
    func_list : list
        Mode shape functions.

    Returns
    -------
    F_hist : ndarray
        Force time history of shape (nt, N_bridge).
        Each row is the force vector at one time step.
    """

    nt = len(t_array)
    F_hist = np.zeros((nt, N_bridge))

    for i, ti in enumerate(t_array):

        # choose pedestrian positions at this time step
        if np.ndim(xrb_over_time) == 2:
            xrb = xrb_over_time[:, i]
        else:
            xrb = xrb_over_time

        _, _, _, F = MatrixAssemblesymetric_socialeeklo(
            case_pedestrian, case_bridge, mped, kped, cped,
            xrb, lb, modalmass, N_bridge, numped, ti, func_list
        )

        F_hist[i, :] = F[:, 0]

    return F_hist

def FirstModalForceTimeHistory(case_pedestrian, mped, xrb, lb, modalmass,
                               numped, t_array, func_list):
    """
    Returns the first modal force time history for fixed pedestrian positions xrb.

    Parameters
    ----------
    case_pedestrian : Pedestrian object
    mped : array-like
        Pedestrian masses
    xrb : array-like, shape (numped,)
        Fixed pedestrian positions
    lb : float
        Bridge length
    modalmass : float
        Modal mass
    numped : int
        Number of pedestrians
    t_array : array-like
        Time vector
    func_list : list
        Mode shape functions; func_list[0] is the first mode

    Returns
    -------
    Q1_t : ndarray, shape (nt,)
        First modal force time history
    """

    nt = len(t_array)
    Q1_t = np.zeros(nt)

    if numped == 0:
        return Q1_t

    # only first mode -> N_bridge = 1
    NN = Phi_matrix_data(xrb, lb, modalmass, 1, func_list)   # shape (1, numped)
    
    for i, ti in enumerate(t_array):
        Ft = pedestrian.calcPedForce(case_pedestrian, ti)     # shape (numped,)
        Q1_t[i] = np.sum(Ft * NN[0, :])

    return Q1_t


def generate_uniform_crowd_xrb(length, width, density, ny=None):
    """
    Generate longitudinal xrb positions for a uniform 2D crowd layout.

    Parameters
    ----------
    length : float
        Bridge length (m)
    width : float
        Bridge width (m)
    density : float
        Crowd density (ped/m^2)
    ny : int or None
        Number of pedestrians across the width.
        If None, chosen automatically.

    Returns
    -------
    xrb : ndarray
        Longitudinal coordinates of all pedestrians
    yrb : ndarray
        Transverse coordinates of all pedestrians
    numped : int
        Total number of pedestrians
    """
    numped = int(round(density * length * width))

    if numped == 0:
        return np.array([]), np.array([]), 0

    if ny is None:
        ny = max(1, int(round(width / 1.0)))   # about 1 m lane spacing

    nx = int(math.ceil(numped / ny))

    x = np.linspace(length/(2*nx), length - length/(2*nx), nx)
    y = np.linspace(width/(2*ny), width - width/(2*ny), ny)

    X, Y = np.meshgrid(x, y)

    xrb = X.ravel()[:numped]
    yrb = Y.ravel()[:numped]

    return xrb, yrb, numped