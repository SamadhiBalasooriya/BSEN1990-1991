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
from matplotlib import pyplot as plt
import timeit

def run_simulation(args):
    seed, peddamp, pedBodyF, beamFreq = args
    if seed is not None:
        np.random.seed(seed)

    # ---------------- Crowd Generator ----------------
    length = 50.0
    width  = 2.0
    hht = 0.01
    meanvelocity = 1.25
    stddev = 0.19

    t = np.arange(0, 100, hht)
    totalTimeSteps = t.size

    tocity  = 10
    outcity = 10

    initial_state = initial_state_corridor(tocity, outcity, length, width, meanvelocity, stddev)

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
        x_coords[i, :] = states_sf[i][:, 0].numpy()

    # ---------------- Beam + pedestrians ----------------
    modalDampingRatio = 0.005
    numbers = 1
    modal_mass = 12500.0
    linearMass = 500.0

    def curve1(x):
        return np.sin(np.pi * x / 50.0)

    pedmass = 73.85
    pace  = 2.0
    pedvelocity = 1.25

    xrb = x_coords
    numped = xrb.shape[1]

    # "off bridge" positions (baseline)
    xrb0 = -1.0 * np.ones(numped)

    # random pace + phase
    pedpace  = np.random.normal(pace, 0.18, numped)
    pedphase = np.random.uniform(0, 2*np.pi, numped)

    # pedestrian body params (constant here)
    Fvalues  = np.full(numped, pedBodyF)
    xivalues = np.full(numped, peddamp)

    kped = (2*np.pi*Fvalues)**2 * pedmass
    cped = (2*np.pi*Fvalues) * 2*xivalues * pedmass

    mped = np.repeat(pedmass, numped)

    func_list = [curve1]
    modalmass = [modal_mass]

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
    ExcitationFrequency = np.arange(1.0, 4.0, 0.005)
    omega = 2 * np.pi * ExcitationFrequency

    # ---------------- BASELINE: bridge H00 from (0,0) ----------------
    M0, K0, C0, _ = MatrixAssemblesymetric_social(
        Human, Bridge, mped, kped, cped,
        xrb0, length, modalmass, numbers, numped, t[0], func_list
    )
    H0 = compute_accelerance(M0, K0, C0, omega)

    H0_bridge = np.abs(H0[:, 0, 0])      # bridge drive-point accelerance
    H00 = H0_bridge.max()
    f00 = ExcitationFrequency[H0_bridge.argmax()]

    # ---------------- outputs: bridge effective series ----------------
    eff_damping_series   = np.zeros(totalTimeSteps)
    eff_frequency_series = np.zeros(totalTimeSteps)

    # ---------------- time loop ----------------
    for i in range(totalTimeSteps):
        M, K, C, _ = MatrixAssemblesymetric_social(
            Human, Bridge, mped, kped, cped,
            xrb[i, :], length, modalmass, numbers, numped, t[i], func_list
        )
        H = compute_accelerance(M, K, C, omega)

        # bridge (0,0)
        H_bridge = np.abs(H[:, 0, 0])
        max_val  = H_bridge.max()
        f_peak   = ExcitationFrequency[H_bridge.argmax()]

        # time-varying modal pedestrian mass (depends on positions)
        xr_i = xrb[i, :]
        modalpedmass_i = pedmass * np.sum(curve1(xr_i)**2)

        # effective damping + frequency (your formula)
        zeta_eff = modalDampingRatio * (H00 / max_val) * (modal_mass / (modal_mass + modalpedmass_i))
        zeta_eff = min(zeta_eff, 0.7)

        f_eff = f_peak * np.sqrt(max(0.0, 1 - 2 * zeta_eff**2))

        eff_damping_series[i]   = zeta_eff
        eff_frequency_series[i] = f_eff

    # mass ratio (for reporting)
    xr_mid = x_coords[totalTimeSteps // 2, :]
    modalpedmass_mid = pedmass * np.sum(curve1(xr_mid)**2)
    mass_ratio = modalpedmass_mid / modal_mass

    return t, eff_damping_series, eff_frequency_series, mass_ratio, numped
