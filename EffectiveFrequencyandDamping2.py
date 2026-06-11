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
        np.random.seed(seed)  # Ensures independent randomness per process
    #_____________________________________________________________Crowd Generator________________________________________________________________________________
    length = 50.0
    width = 2.0
    hht = 0.01
    meanvelocity = 1.25
    stddev = 0.19

    t = np.arange(0, 100, hht)
    totalTimeSteps = np.size(t)
    print(np.size(t))

    tocity = 20 #bounded_gamma_sample1(16.7,12.0,0.0,67.0)
    print(tocity)
    outcity = 20 #bounded_gamma_sample1(9.4,4.1,1.0,21.0)
    print(outcity)

    initial_state = initial_state_corridor(tocity,outcity,-1.0,1.5,meanvelocity,stddev)


    upper_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), width)], -1)
    lower_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), 0)], -1)
    ped_space = socialforce.potentials.PedSpacePotential([upper_wall, lower_wall])

    #ped_ped = socialforce.potentials.PedPedPotential()
    #ped_ped = socialforce.potentials.PedPedPotentialDiamond(sigma=0.5)
    ped_ped = socialforce.potentials.PedPedPotentialDiamond(sigma=0.5, asymmetry_angle=-20.0)
    #ped_ped = socialforce.potentials.PedPedPotential2D()


    simulator = socialforce.Simulator(ped_ped=ped_ped, ped_space=ped_space,
                                  oversampling=2, delta_t=hht)
    simulator.integrator = socialforce.simulator.PeriodicBoundary(
        simulator.integrator, x_boundary=[0, +length])

    with torch.no_grad():
            states_sf = simulator.run(initial_state, totalTimeSteps)
 

    #__________________________________________________________________Extracting the coordinates_______________________________________________________________________

    # Extract the number of pedestrians and timesteps
    num_timesteps = len(states_sf)
    #print(states_sf.size())
    num_pedestrians = states_sf[0].shape[0]
    #print("Number of pedestrians:", num_pedestrians)
    #print("Number of timesteps:", num_timesteps)

    # Create an array to store the variations in coordinates
    x_coords = np.zeros((num_timesteps, num_pedestrians))
    y_coords = np.zeros((num_timesteps, num_pedestrians))

    # Fill in the matrix
    for i in range(num_timesteps):
        x_coords[i, :] = states_sf[i][:, 0].numpy()  # Extract x-coordinates
        y_coords[i, :] = states_sf[i][:, 1].numpy()  # Extract y-coordinates

    #__________________________________________________________________end of Extracting the coordinates_______________________________________________________________________



    #_______________________________________________________________solver_______________________________________________________________________


    #step 1 setup beam and pedestrians
    #beam

    height = 0.6  # h - Height (m)
    E = 200e9  # E - Young's modulus (N/m^2)
    modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
    nHigh = 3  # nHigh - Higher mode for damping matrix
    beamFreq =beamFreq #Hz
    area = 0.3162  # A - Cross-section area (m^2)
    linearMass = 500  # m - Linear mass (kg/m)    --here it need to be derived
    x_interested= length/2
    numbers = 1       #modes

    numbers = 1 #modes being considered
    modal_mass = 12500 #kg modal mass
    
    def curve1(x): #modeshape
        return  np.sin(np.pi*x/50)

    #single pedestrian properties


    pedmass = 73.85     #kg mean
    #peddamp = 0.3    
    pace  = 2.0     #Hz
    pedvelocity = 1.25
    #pedBodyF= 3.0 #Hz

    xr_case1 = x_coords[totalTimeSteps//2, :]
    modalpedmass_1 = pedmass * np.sum((curve1(xr_case1))**2)

    mass_ratio = modalpedmass_1 / modal_mass
    print(f"Mass ratio: {mass_ratio:.4f}")

    #generating random pace and phase for crowd
    xrb = x_coords
    numped = xrb.shape[1] 
    xrb0 = -1 * np.ones(numped)
    print(numped)
    d = numped/(length*width) 
    pedpace = np.random.normal(pace, 0.18, numped)
    pedphase = np.random.uniform(0, 2*np.pi, numped)

    #ped body damp and stiff
    Fvalues = np.full(numped, pedBodyF) #np.random.normal(pedBodyF, 0.18, numped)
    xivalues = np.full(numped, peddamp) #np.random.normal(peddamp, 0.03, numped)
    kped1=(2*np.pi*Fvalues)**2*pedmass
    cped1 = (2*np.pi*Fvalues)*2*xivalues*pedmass




    #inout pedestrian mass,damping and stiffneess arrays
    mped = np.repeat(pedmass,numped)  #np.array([pedmass])
    cped = cped1 #np.repeat(cped1,numped)    #np.array([cped1])
    kped = kped1 #np.repeat(kped1,numped)

    func_list=[curve1] #list of functions for mode shapes
    modalmass = [modal_mass]

    #bridge stiffness
    modulus =linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2  #E*(width*height**3)/12




    Bridge = bridge(   
        length = length,                 # m
        modulus = modulus,               # N m^2
        density = linearMass,            # kg/m
        damp    = modalDampingRatio ,    #%
        numbers = numbers,                    #modes
        freq = beamFreq )

    Human = Pedestrian(
            mass = mped,     #kg
             damp = cped ,   #%
             stiff = kped, #N/m
             pace  = pedpace ,    #Hz
             phase = pedphase,
             location = xrb,
             velocity = pedvelocity,
         
             iSync=0.3)

    eff_damping_all  = np.zeros((totalTimeSteps, numped))
    eff_frequency_all = np.zeros((totalTimeSteps, numped))

    ExcitationFrequency = np.arange(1.0, 4.0 , 0.005)  # Hz
    omega = 2 * np.pi * ExcitationFrequency  # Convert frequency to angular frequency

    M0, K0, C0, _ = MatrixAssemblesymetric_social(Human, Bridge, mped, kped, cped, xrb0, length, modalmass, numbers, numped, t[0], func_list)
    H0 = compute_accelerance(M0, K0, C0, omega)
    # baseline peaks for each pedestrian DOF (index 1..numped)
    H00 = np.zeros(numped)
    f00 = np.zeros(numped)
    for j in range(numped):
        idx = 1 + j
        Hj0 = np.abs(H0[:, idx, idx])
        H00[j] = Hj0.max()
        f00[j]  = ExcitationFrequency[Hj0.argmax()]
# --- time loop
    for i in range(totalTimeSteps):
        M, K, C, _ = MatrixAssemblesymetric_social(
            Human, Bridge, mped, kped, cped, xrb[i, :], length, modalmass, numbers, numped, t[i], func_list
        )
        H = compute_accelerance(M, K, C, omega)
        for j in range(numped):
            idx = 1 + j  # DOF of pedestrian j

            Hj = np.abs(H[:, idx, idx])
            max_val = Hj.max()
            f_peak  = ExcitationFrequency[Hj.argmax()]

            # your current "effective damping" definition
            zeta_eff = peddamp * (H00[j] / max_val)

            # guard: avoid sqrt of negative
            zeta_eff = min(zeta_eff, 0.7)  # just to prevent nonsense; tweak if you want

            f_eff = f_peak * np.sqrt(max(0.0, 1 - 2 * zeta_eff**2))

            eff_damping_all[i, j]  = zeta_eff
            eff_frequency_all[i, j] = f_eff

    return t, eff_damping_all, eff_frequency_all, mass_ratio, numped