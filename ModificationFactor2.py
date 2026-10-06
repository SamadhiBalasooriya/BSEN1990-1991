import numpy as np
from IPython.display import HTML, display
import numpy as np
from EquivalentFreqandDamp import compute_accelerance
from matrix import*  
from solver import *
from pedestrian import* 



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

    # -----------------------------
    # MODAL PEDESTRIAN MASS RATIO
    # -----------------------------
    xr_i = xrb

    modalpedmass_i = pedmass * np.sum(curve1(xr_i)**2) / modal_mass

    # code rule: include pedestrian modal mass only when > 5%
    if modalpedmass_i > 0.05:
        code_mass_factor = 1.0 + modalpedmass_i
    else:
        code_mass_factor = 1.0

    modal_mass_code = modal_mass * code_mass_factor
    inearMass_code = linearMass * code_mass_factor
    modalmass_code = [modal_mass_code]   
    
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

    idx0_peak = np.argmax(H0_bridge)

    H0_max = H0_bridge[idx0_peak]
    f0_peak = ExcitationFrequency[idx0_peak]   # Hz
    #w0_peak = omega[idx0_peak]                 # rad/s

        
    M, K, C, _ = MatrixAssemblesymetric_socialeeklo(
        Human, Bridge, mped, kped, cped,
        xrb, length, modalmass, numbers, numped, t[0], func_list
    )
    H = compute_accelerance(M, K, C, omega)
    H_bridge = np.abs(H[:, 0, 0])

    idx_eff_peak = np.argmax(H_bridge)

    H_max = H_bridge[idx_eff_peak]
    f_eff_peak = ExcitationFrequency[idx_eff_peak]   # Hz
    #w_eff_peak = omega[idx_eff_peak]                 # rad/s

    # time-varying modal pedestrian mass (depends on positions)
    xr_i = xrb
    modalpedmass_i = pedmass * np.sum(curve1(xr_i)**2)/modal_mass

    # ratios
    frf_ratio = H_max / H0_max

    # psi values using psi_w_h1 imported from pedestrian.py
    psi_0 = float(psi_w_h1(f0_peak))   # psi at uncoupled peak frequency
    psi_eff = float(psi_w_h1(f_eff_peak))  # psi at coupled/effective peak frequency

    # psi ratio:
    # psi(f_eff) / psi(f0)
    eps = 1e-12

    # Only use psi ratio when both frequencies have non-zero psi values
    if (psi_0 > eps) and (psi_eff > eps):
        psi_ratio = psi_eff / psi_0
        psi_ratio_valid = True
    else:
        psi_ratio = 0.0
        psi_ratio_valid = False

    # Option 1: FRF effect is always active, psi correction only active when valid
    if psi_ratio_valid:
        modification_factor = frf_ratio * psi_ratio
    else:
        modification_factor = 0

    
    
    return (
    t,
    modalpedmass_i,
    modification_factor,
    frf_ratio,
    psi_ratio,
    psi_ratio_valid,
    f_eff_peak,
    f0_peak
)