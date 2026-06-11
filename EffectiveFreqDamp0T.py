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
import timeit

from trialUDL import curve1, curve2


def run_simulation(args):
    seed, peddamp, pedBodyF, beamFreq, beamdamp, Tocity, Outcity,Length, Width = args
    if seed is not None:
        np.random.seed(seed)

    # ---------------- Crowd Generator ----------------
    length = Length
    width  = Width
    hht = 0.01
    meanvelocity = 1.25
    stddev = 0.03

    t = np.arange(0, 100, hht)
    totalTimeSteps = t.size

    tocity  = Tocity
    outcity = Outcity

    initial_state = initial_state_corridor(tocity, outcity, length, width, meanvelocity, stddev)

    '''
    upper_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), width)], -1)
    lower_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), 0)], -1)
    ped_space = socialforce.potentials.PedSpacePotential([upper_wall, lower_wall])

    ped_ped = socialforce.potentials.PedPedPotentialDiamond(sigma=0.5, asymmetry_angle=-20.0)

    simulator = socialforce.Simulator(ped_ped=ped_ped, ped_space=ped_space,
                                      oversampling=2, delta_t=hht)
    simulator.integrator = socialforce.simulator.PeriodicBoundary(
        simulator.integrator, x_boundary=[0, +length])

    with torch.no_grad():
        states_sf = simulator.run(initial_state, totalTimeSteps)

    # ---------------- Extract x coordinates ----------------
    num_timesteps   = len(states_sf)
    num_pedestrians = states_sf[0].shape[0]

    x_coords = np.zeros((num_timesteps, num_pedestrians))
    for i in range(num_timesteps):
        x_coords[i, :] = states_sf[i][:, 0].numpy()'''

    initial_x = initial_state[:, 0].numpy()
    
    # ---------------- Beam + pedestrians ----------------
    modalDampingRatio = beamdamp
    numbers = 1
    #modal_mass = 22000.0
    #linearMass = 1020.0

    '''def curve1(x):
        return (1.71144429e-14 * x**8
        - 6.76029672e-12 * x**7
        + 1.05320380e-09 * x**6
        - 8.14053964e-08 * x**5
        + 3.27033830e-06 * x**4
        - 7.15713216e-05 * x**3
        + 7.95784053e-04 * x**2
        + 2.20681765e-02 * x
        + 2.26628021e-03)


    def curve1(x):
        return (
        -8.02321495e-16 * x**9
        + 4.32499733e-13 * x**8
        - 9.53553597e-11 * x**7
        + 1.09530960e-08 * x**6
        - 6.90120316e-07 * x**5
        + 2.30469265e-05 * x**4
        - 3.76429912e-04 * x**3
        + 3.46209330e-03 * x**2
        - 1.90877156e-02 * x
        + 3.95334984e-04
    )'''

    linearMass = 500.0
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

    xrb = initial_x
    numped = tocity + outcity
    # "off bridge" positions (baseline)
    xrb0 = -1.0 * np.ones(numped)

    # random pace + phase
    pedpace  = np.random.normal(1.69, 0.18, numped)
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
    ExcitationFrequency = np.arange(1.0, 4.0, 0.05)
    omega = 2 * np.pi * ExcitationFrequency

    # ---------------- BASELINE: bridge H00 from (0,0) ----------------
    M0, K0, C0, _ = MatrixAssemblesymetric_socialeeklo(
        Human, Bridge, mped, kped, cped,
        xrb0, length, modalmass, numbers, numped, t[0], func_list
    )
    H0 = compute_accelerance(M0, K0, C0, omega)

    H0_bridge = np.abs(H0[:, 0, 0])      # bridge drive-point accelerance
    H00 = H0_bridge.max()
    f00 = ExcitationFrequency[H0_bridge.argmax()]

        
    M, K, C, _ = MatrixAssemblesymetric_socialeeklo(
        Human, Bridge, mped, kped, cped,
        xrb, length, modalmass, numbers, numped, t[0], func_list
    )
    H = compute_accelerance(M, K, C, omega)

    # bridge (0,0)
    H_bridge = np.abs(H[:, 0, 0])
    max_val  = H_bridge.max()
    f_peak   = ExcitationFrequency[H_bridge.argmax()]

    # time-varying modal pedestrian mass (depends on positions)
    xr_i = xrb
    modalpedmass_i = pedmass * np.sum(curve1(xr_i)**2)

    # effective damping + frequency (your formula)
    zeta_eff = modalDampingRatio * (H00 / max_val) * (modal_mass / (modal_mass + modalpedmass_i))
    zeta_eff = min(zeta_eff, 0.7)

    f_eff = f_peak * np.sqrt(max(0.0, 1 - 2 * zeta_eff**2))

    eff_damping_series   = zeta_eff
    eff_frequency_series = f_eff

    # mass ratio (for reporting)
    xr_mid = xrb
    modalpedmass_mid = pedmass * np.sum(curve1(xr_mid)**2)
    mass_ratio = modalpedmass_mid / modal_mass

    return t, eff_damping_series, eff_frequency_series, mass_ratio, numped