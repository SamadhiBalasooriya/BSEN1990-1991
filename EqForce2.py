import numpy as np
import socialforce
import imageio
from IPython.display import HTML, display
import numpy as np
import torch
from socialforcefunctions import initial_state_corridor,bounded_gamma_sample1
from EquivalentFreqandDamp import compute_accelerance, modal_step, track_modes
import pandas as pd
from matrix import*  
from solver import *
from pedestrian import* 
#from matplotlib import pyplot as plt
from scipy.integrate import quad
import timeit

from trialUDL import curve1, curve2


def run_simulation(args):
    seed, peddamp, pedBodyF, beamFreq, beamdamp, density,Length, Width, Linearmass = args
    if seed is not None:
        np.random.seed(seed)

    # ---------------- Crowd Generator ----------------
    length = Length
    width  = Width
    hht = 0.01
    meanvelocity = 1.25
    stddev = 0.03

    t = np.arange(0, 20, hht)
    totalTimeSteps = t.size

    xrb, yrb, numped = generate_uniform_crowd_xrb(length, width, density)
    
    # ---------------- Beam + pedestrians ----------------
    modalDampingRatio = beamdamp
    numbers = 1


    linearMass = Linearmass
    modal_mass = linearMass*length/2 #kg modal mass
    #def curve1(x): #modeshape
    #    return  1.234567901234568e-06*x**4 + -0.00014814814814814815*x**3 + 0.0044444444444444444*x**2

    def curve1(x):
        return np.sin(np.pi*x/length)

    def curve2(x):
        return np.sin(2*np.pi*x/length)

    func_list=[curve1] #list of functions for mode shapes
    modalmass = [modal_mass] #list of modal mass for each mode shape



    pedmass = 73.85
    pace  = 2.0
    pedvelocity = 1.10

    
    
    # "off bridge" positions (baseline)
    xrb0 = -1.0 * np.ones(numped)

    # random pace + phase
    pedpace  = np.random.normal(2.0, 0.18, numped)
    pedphase = np.random.uniform(0, 2*np.pi, numped)

    # pedestrian body params (constant here)
    Fvalues  = np.full(numped, pedBodyF)
    xivalues = np.full(numped, peddamp)

    kped = (2*np.pi*Fvalues)**2 * pedmass
    cped = (2*np.pi*Fvalues) * 2*xivalues * pedmass

    mped = np.repeat(pedmass, numped)

    # bridge stiffness (your formula)
    modulus = linearMass * ((2 * np.pi * beamFreq) * (np.pi / length) ** (-2)) ** 2

    Bridge = bridge(
        length  = length,
        modulus = modulus,
        density = linearMass,
        damp    = modalDampingRatio,
        numbers = numbers,
        freq    = beamFreq
    )

    Human = Pedestrian(
        mass     = mped,
        damp     = cped,
        stiff    = kped,
        pace     = pedpace,
        phase    = pedphase,
        location = xrb,
        velocity = pedvelocity,
        iSync    = 0.3
    )

    # ---------------- Frequency grid ----------------
    #ExcitationFrequency = np.arange(0.1, 6.0, 0.001)
    if beamFreq > 6:
        ExcitationFrequency = np.arange(beamFreq - 2.9, beamFreq + 3.0, 0.001)
    else:
        ExcitationFrequency = np.arange(0.05, 6.0, 0.001)

    omega = 2 * np.pi * ExcitationFrequency

    # ---------------- BASELINE: bridge H00 from (0,0) ----------------
    M0, K0, C0, _ = MatrixAssemblesymetric_socialeeklo(
        Human, Bridge, mped, kped, cped,
        xrb0, length, modalmass, numbers, numped, t[0], func_list
    )
    H0 = compute_accelerance(M0, K0, C0, omega)

    H0_bridge = np.abs(H0[:, 0, 0])      # bridge drive-point accelerance
    H0_max = np.max(H0_bridge)
    f0_peak = ExcitationFrequency[np.argmax(H0_bridge)]

        
    M, K, C, _ = MatrixAssemblesymetric_socialeeklo(
        Human, Bridge, mped, kped, cped,
        xrb, length, modalmass, numbers, numped, t[0], func_list
    )
    H = compute_accelerance(M, K, C, omega)
    H_bridge = np.abs(H[:, 0, 0])
    H_max = np.max(H_bridge)
    f_peak = ExcitationFrequency[np.argmax(H_bridge)]

    # time-varying modal pedestrian mass (depends on positions)
    xr_i = xrb
    modalpedmass_i = pedmass * np.sum(curve1(xr_i)**2)/modal_mass

    # ratios
    accel_ratio = H_max / H0_max          # current / reference
    #accel_ratio_inv = H0_max / H_max      # reference / current

    '''
    # ---------------- FIRST MODAL FORCE TIME HISTORY ----------------
    #Q1_t = np.abs(FirstModalForceTimeHistory(Human, mped, xrb, length, modalmass, numped, t, func_list))
    Q1_t = FirstModalForceTimeHistory(Human, mped, xrb, length, modalmass, numped, t, func_list)
    # equivalent first modal force using peak-FRF ratio
    Q1_eq = accel_ratio * Q1_t

    alpha_udl, _ = quad(curve1, 0.0, length)

    # convert mass-normalized modal force to raw modal force
    Q1_eq_raw = np.sqrt(modal_mass) * Q1_eq
    Q1_t_raw = np.sqrt(modal_mass) * Q1_t
    eq_UDL = Q1_eq_raw / (alpha_udl*width)
    realUDL = Q1_t_raw / (alpha_udl*width)

    # effective damping + frequency (your formula)
    zeta_eff = modalDampingRatio * (H00 / max_val) * (modal_mass / (modal_mass + modalpedmass_i))
    zeta_eff = min(zeta_eff, 0.7)

    f_eff = f_peak * np.sqrt(max(0.0, 1 - 2 * zeta_eff**2))

    eff_damping_series   = zeta_eff
    eff_frequency_series = f_eff'''


    return t, modalpedmass_i, accel_ratio