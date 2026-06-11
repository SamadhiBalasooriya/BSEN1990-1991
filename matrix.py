import math
from networkx import omega
import numpy as np
from sympy import *
"""
Bridge matrices definition
"""
class bridge:
    """
    Class definition
    """
    def __init__(self, length, modulus, density, damp, numbers, freq):
        self.L = length
        self.EI = modulus
        self.rho = density
        self.damp = damp
        self.n = numbers
        self.freq = freq

    """
    Construct a simply supported beam by superposition method
    Parameters
    ----------
    length : Span length
        Unit m.
    modulous : flexural rigidities
        Unit Nm^2.
    density : unit density
        Unit kg/m.
    damp : damping percentage
        example 0.003 means 0.3%.
    numbers : number of mode shapes considered
        first 10 modes could be sufficient.
    Returns
    -------
    None.
    """
    def Mass_matrix(self):
        """
        Returns the mass matrix which is normalized for the bridge.
        -------
        None.
        """
        mass = [1 for i in range(self.n)]
        mass = np.diag(mass)
        return mass

    def Stiffness_matrix(self):
        """
        Returns normalized stiffness matrix of bridge.
        -------
        None.
        """
        k = []
        for i in range(self.n):
            it = i + 1
            k.append(self.EI / self.rho * (it * np.pi / self.L) ** 4)
        k = np.diag(k)
        return k
    
    def Stiffness_matrix2(self):
        """
        Returns normalized stiffness matrix of bridge when the frequencies of the modes are given.
        -------
        None.
        """
        
        freq_term = (np.pi / self.L) ** 4 #is it here needed
        if np.isscalar(self.EI):
            EI_array = np.full(self.n, self.EI)
        else:
            EI_array = np.array(self.EI)
        k_diag = EI_array / self.rho * freq_term
        return np.diag(k_diag)

    def Stiffness_matrix3(self):
        omega = 2*np.pi*np.asarray(self.freq)
        return np.diag(omega**2)

    def Damp_matrix(self):
        """
        Returns rayleigh damping matrix.
        -------
        None.
        """

        M = bridge.Mass_matrix(self)
        K = bridge.Stiffness_matrix2(self)
    
        if self.n == 1:  # Check if it's a 1DOF system
        # For a 1DOF system, return a constant damping value.
        # You can define what the constant value should be.
            c = 2 * self.damp * (M[0, 0] * K[0, 0] )** 0.5

        else:  # 2DOF or higher

        
            w1 = K[0, 0] ** 0.5
            w2 = K[1, 1] ** 0.5
            a1 = 2 * self.damp * w1 * w2 / (w1 + w2)
            a2 = 2 * self.damp / (w1 + w2)
            c = a1 * M + a2 * K
        return c 
    
    def Damp_matrix2(self):
        M = bridge.Mass_matrix(self)
        K = bridge.Stiffness_matrix2(self)

        c = np.zeros_like(M)  # initialize damping matrix with zeros

        if np.isscalar(self.damp):
            # If damp is a single value, expand it to a list
            damp_array = np.full(self.n, self.damp)
        else:
            damp_array = np.array(self.damp)

        for i in range(self.n):
            w_i = np.sqrt(K[i, i])  # natural frequency for mode i
            # Rayleigh damping for each mode separately:
            a1_i = 2 * damp_array[i] * w_i / (2 * w_i)  # simplifies to just damp[i]
            a2_i = 2 * damp_array[i] / (2 * w_i)        # ~ damp[i]/w_i
        
            c[i, i] = a1_i * M[i, i] + a2_i * K[i, i]

        return c
    
    def Damp_matrix3(self):
        """
        Returns rayleigh damping matrix.
        -------
        None.
        """

        M = bridge.Mass_matrix(self)
        K = bridge.Stiffness_matrix3(self)
    
        if self.n == 1:  # Check if it's a 1DOF system
        # For a 1DOF system, return a constant damping value.
        # You can define what the constant value should be.
            c = 2 * self.damp * (M[0, 0] * K[0, 0] )** 0.5

        else:  # 2DOF or higher

        
            w1 = K[0, 0] ** 0.5
            w2 = K[1, 1] ** 0.5
            a1 = 2 * self.damp * w1 * w2 / (w1 + w2)
            a2 = 2 * self.damp / (w1 + w2)
            c = a1 * M + a2 * K
        return c 
    
    def Damp_matrix4(self):

        K = bridge.Stiffness_matrix3(self)

        # natural frequencies
        omega = np.sqrt(np.diag(K))

        # modal damping ratios
        xi = np.array(self.damp)

        # modal damping matrix
        C = np.diag(2 * xi * omega)

        return C