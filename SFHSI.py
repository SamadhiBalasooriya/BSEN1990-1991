import socialforce
import imageio
from IPython.display import HTML, display
import numpy as np
import torch
from socialforcefunctions import initial_state_corridor,bounded_gamma_sample1
import pandas as pd
from matrix import*  
from solver import *
from pedestrian import* 
from matplotlib import pyplot as plt
import timeit



def run_simulation(args):
    seed, peddamp, pedBodyF, Tocity, Outcity, Length, Width = args
    if seed is not None:
        np.random.seed(seed)  # Ensures independent randomness per process
    #_____________________________________________________________Crowd Generator________________________________________________________________________________
    length = Length #50.0
    width = Width #2.0
    hht =0.01
    meanvelocity = 1.38
    stddev = 0.19

    t =np.arange(0, (length+88) / meanvelocity, hht)
    totalTimeSteps = np.size(t)
    print(np.size(t))

    tocity = Tocity #bounded_gamma_sample1(16.7,12.0,0.0,67.0)
    print(tocity)
    outcity = Outcity #bounded_gamma_sample1(9.4,4.1,1.0,21.0)
    print(outcity)


    initial_state = initial_state_corridor(tocity,outcity,length,width,meanvelocity,stddev)
   


    upper_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), width)], -1)
    lower_wall = torch.stack([torch.linspace(0, length, 1000), torch.full((1000,), 0)], -1)
    ped_space = socialforce.potentials.PedSpacePotential([upper_wall, lower_wall])

    #ped_ped = socialforce.potentials.PedPedPotential()
    #ped_ped = socialforce.potentials.PedPedPotentialDiamond(sigma=0.5)
    ped_ped = socialforce.potentials.PedPedPotentialDiamond(sigma=0.5, asymmetry_angle=-20.0)
    #ped_ped = socialforce.potentials.PedPedPotential2D()


    simulator = socialforce.Simulator(ped_ped=ped_ped, ped_space=ped_space,
                                  oversampling=1, delta_t=hht)
    simulator.integrator = socialforce.simulator.PeriodicBoundary(
        simulator.integrator, x_boundary=[0, +length])

    with torch.no_grad():
        states_sf = simulator.run(initial_state, totalTimeSteps)
    '''
    with socialforce.show.track_canvas(ncols=2, figsize=(12, 2), tight_layout=False) as (ax1, ax2):
    socialforce.show.states(ax1, states_sf[0:1], monochrome=True)
    socialforce.show.space(ax1, ped_space)
    ax1.text(0.1, 0.1, '$t = 0s$', transform=ax1.transAxes)
    ax1.set_xlim(0, +length)

    socialforce.show.states(ax2, states_sf[249:250], monochrome=True)
    socialforce.show.space(ax2, ped_space)
    ax2.text(0.1, 0.1, '$t = 20s$', transform=ax2.transAxes)
    ax2.set_xlim(0, length)'''
    #print(states_sf[0:1])
    #   print(states_sf[249:250])
    #__________________________________________________________________end of crowd generator_________________________________________________________________________




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
    for t in range(num_timesteps):
        x_coords[t, :] = states_sf[t][:, 0].numpy()  # Extract x-coordinates
        y_coords[t, :] = states_sf[t][:, 1].numpy()  # Extract y-coordinates

    #__________________________________________________________________end of Extracting the coordinates_______________________________________________________________________



    #_______________________________________________________________solver_______________________________________________________________________


    #step 1 setup beam and pedestrians
    #beam

    height = 0.6  # h - Height (m)
    E = 200e9  # E - Young's modulus (N/m^2)
    modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
    nHigh = 3  # nHigh - Higher mode for damping matrix
    beamFreq = np.array([2,8]) #Hz
    area = 0.3162  # A - Cross-section area (m^2)
    linearMass = 500.0  # m - Linear mass (kg/m)    --here it need to be derived
    x_interested= length/2
    numbers = 2       #modes

    numbers = 2 #modes being considered
    modal_mass = linearMass*length/2 #kg modal mass
    #def curve1(x): #modeshape
    #    return  1.234567901234568e-06*x**4 + -0.00014814814814814815*x**3 + 0.0044444444444444444*x**2

    def curve1(x):
        return np.sin(np.pi*x/length)

    def curve2(x):
        return np.sin(2*np.pi*x/length)

    
    #single pedestrian properties
    pedmass = 700/9.81     #kg mean
    #peddamp = 0.3    
    pace  = 2.00     #Hz
    pedvelocity = 1.38
    #pedBodyF= 3.0 #Hz



    #generating random pace and phase for crowd
    xrb = x_coords
    numped = xrb.shape[1] 
    print(numped)
    d = numped/(length*width) 
    
    
    pedpace = np.random.normal(pace, 0.18, numped)
    #pedphase = np.random.uniform(0, 2*np.pi, numped)
    pedphase= np.repeat(0,numped)

    Ns = int(0.3*numped) #int(np.sqrt(numped))

    syc_idx = np.random.choice(numped,Ns,replace=False)

    #pedphase[syc_idx] = 0.0
    pedpace[syc_idx] = beamFreq[0]

    #ped body damp and stiff
    Fvalues = np.random.normal(pedBodyF, 0.18, numped)
    xivalues = np.random.normal(peddamp, 0.03, numped)
    kped1=(2*np.pi*Fvalues)**2*pedmass
    cped1 = (2*np.pi*Fvalues)*2*xivalues*pedmass

    '''
    Sc = 0.3                                         #PyHSI iSync
    print(numped)
    if 0 < Sc <= 1:
        Ns = int(np.floor(numped * Sc))             # Number of synchronized individuals
        rp = np.random.permutation(numped)          # Random shuffle of indices
        iSync = np.sort(rp[:Ns])                    # Indices to synchronize

        # Assign the same pacing frequency and phase
        sPace = beamFreq #np.random.normal(2, 0.17)
        pedpace[iSync] = sPace

        sPhase = 0
        pedphase[iSync] = sPhase

        print("\nSynchronized indices:", iSync)
        print("Shared pace value:", round(sPace, 2))
        print("Shared phase value:", round(sPhase, 2))

    print("\nUpdated pPace:", np.round(pedpace, 2))
    print("Updated pPhase:", np.round(pedphase, 2))'''

    ## Target mean and std deviation for pedestrian masses
    M = 73.85  # Mean mass in kg
    S = 15.68  # Std deviation in kg

    # Convert to lognormal parameters
    mu = np.log(M**2 / np.sqrt(S**2 + M**2))
    sigma = np.sqrt(np.log(1 + (S**2 / M**2)))


    randompedmass = np.random.lognormal(mu, sigma, numped)



    #inout pedestrian mass,damping and stiffneess arrays
    mped = randompedmass  #np.repeat(pedmass,numped)  #np.array([pedmass])
    cped = cped1 #np.repeat(cped1,numped)    #np.array([cped1])
    kped = kped1 #np.repeat(kped1,numped)

    func_list=[curve1, curve2] #list of functions for mode shapes
    modalmass = [modal_mass, modal_mass] #list of modal mass for each mode shape

    #bridge stiffness
    modulus =linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2  #E*(width*height**3)/12




    Bridge = bridge(   
        length = length,                 # m
        modulus = modulus,               # N m^2
        density = linearMass,            # kg/m
        damp    = modalDampingRatio ,    #%
        numbers = numbers,
        freq= beamFreq  )                   #modes


    Human = Pedestrian(
            mass = mped,     #kg
             damp = cped ,   #%
             stiff = kped, #N/m
             pace  = pedpace ,    #Hz
             phase = pedphase,
             location = xrb,
             velocity = pedvelocity,
         
             iSync=0.3)

    _,_,ddu_hsi = Newmarksuper_HSIsocial(Human,Bridge,numped,numbers,length,hht,pedvelocity,mped,kped,cped,xrb,modalmass,func_list)
    print("complete")
    accn_hsi = accdyn_super_social(Bridge,ddu_hsi,x_interested,modalmass,func_list)
    #vertical_displacement = accdyn_super(Bridge,u,25,hht)

    # Apply first mode shape to every pedestrian position at every time step
    phi1_all = curve1(xrb)   # same shape as xrb

    # Modal pedestrian mass contribution at each time step
    # shape: (num_timesteps,)
    modalpedmass_1_all = pedmass * np.sum(phi1_all**2, axis=1)

    # Mass ratio at each time step
    mass_ratio_all = modalpedmass_1_all / modal_mass

    # Mean mass ratio over all time steps
    mean_mass_ratio = np.mean(mass_ratio_all)
    
    return accn_hsi,mean_mass_ratio

'''
#absolute_max = max(xrb, key=abs)
t =np.arange(0, (length + 5) / pedvelocity, hht)
#plt.plot(t,accn , label ="without HSI" ,color='r')
plt.plot(t,accn_hsi,label ="with HSI",color='b')
#plt.plot(t,accn_EQ,label ="with EQ",color='g')
#plt.plot(t, accn, label='with MF',color='r')
#plt.plot(6.97, 2.81, 'ro', label='Predicted peak using constant equivalent FRF')
plt.title("mid span acceleration")
plt.xlabel("time(s)")
plt.ylabel("m/s2")
plt.legend()
plt.tight_layout()
plt.savefig("All In one", format='pdf', dpi=300)  
plt.show()
#print("ddu",ddu)
#print("accn",accn)'''