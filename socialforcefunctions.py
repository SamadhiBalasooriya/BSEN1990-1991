import math
import numpy as np
import scipy
from scipy.linalg import eig
from scipy.linalg import eigh
from matrix import*
import matrix
from pedestrian import*
from sympy import *
from matplotlib import pyplot as plt
import IPython
import numpy as np
import torch
import socialforce

import pedestrian



def initial_state_corridor(tocity,outcity,length,width,meanvelocity, standarddev):
    #_ = torch.manual_seed(42) #for checking perposes remove# to make it non random

    # first n people go right, second n people go left
    state = torch.zeros((tocity+outcity, 6))

    # initial positions
    #state[:tocity, 0:2] = (torch.rand((tocity, 2))) * torch.tensor([length, (width-1)]) 
    #state[tocity:, 0:2] = (torch.rand((outcity, 2))) * torch.tensor([length, (width-1)]) 

    # initial positions
    state[:tocity, 0] = torch.rand(tocity) * length                        # x in [0, length)
    state[:tocity, 1] = torch.rand(tocity) * (width - 1) + 0.5              # y in [1, width - 1)

    state[tocity:, 0] = torch.rand(outcity) * length
    state[tocity:, 1] = torch.rand(outcity) * (width - 1) + 0.5

    # velocityin x direction
    state[:tocity, 2] = torch.normal(torch.full((tocity,), meanvelocity), standarddev)
    state[tocity:, 2] = torch.normal(torch.full((outcity,), -meanvelocity), standarddev)

    # x destination
    state[:tocity, 4] = 100+length
    state[tocity:, 4] = -100

    #Y destination
    #state[:n, 5] = width / 2
    #state[n:, 5] = width / 2

    return state

def initial_state_corridor_for_1(tocity, outcity, length, width, meanvelocity, standarddev):
    state = torch.zeros((tocity + outcity, 6))

    # ----- to-city people -----
    # x = 0 for to-city group (start at left boundary)
    state[:tocity, 0] = 0.0
    # y as before: [0.5, width-0.5]
    state[:tocity, 1] = torch.rand(tocity) * (width - 1) + 0.5

    # ----- out-of-city people (if any) -----
    state[tocity:, 0] = length
    state[tocity:, 1] = torch.rand(outcity) * (width - 1) + 0.5

    # velocities
    state[:tocity, 2] = torch.normal(torch.full((tocity,), meanvelocity), standarddev)
    state[tocity:, 2] = torch.normal(torch.full((outcity,), -meanvelocity), standarddev)

    # x destinations
    state[:tocity, 4] = length + 100.0
    state[tocity:, 4] = -100.0

    return state


def bounded_gamma_samples(mean, std_dev, min_val, max_val, size):
    # Convert to shape and scale
    shape = (mean / std_dev) ** 2
    scale = (std_dev ** 2) / mean

    samples = []
    while len(samples) < size:
        s = np.random.gamma(shape, scale)
        if min_val <= s <= max_val:
            samples.append(s)

    arr = np.array(samples)
    rounded = np.round(arr).astype(int)
    return rounded

import numpy as np

def bounded_gamma_sample1(mean, std_dev, min_val, max_val):
    # Convert to shape and scale
    shape = (mean / std_dev) ** 2
    scale = (std_dev ** 2) / mean

    while True:
        s = np.random.gamma(shape, scale)
        if min_val <= s <= max_val:
            return int(np.round(s))
        

def initial_state_corridor_clusters(tocity,outcity,length,width,meanvelocity, standarddev,clusters2,clusters3):
    #_ = torch.manual_seed(42) #for checking perposes remove# to make it non random

    # first n people go right, second n people go left
    state = torch.zeros((tocity+outcity, 6))

    # initial positions
    #state[:tocity, 0:2] = (torch.rand((tocity, 2))) * torch.tensor([length, (width-1)]) 
    #state[tocity:, 0:2] = (torch.rand((outcity, 2))) * torch.tensor([length, (width-1)]) 

    # initial positions
    state[:tocity, 0] = torch.rand(tocity) * length                        # x in [0, length)
    state[:tocity, 1] = torch.rand(tocity) * (width - 1) + 0.5              # y in [0.5, width - 0.5)

    state[tocity:, 0] = torch.rand(outcity) * length
    state[tocity:, 1] = torch.rand(outcity) * (width - 1) + 0.5

    # velocityin x direction
    state[:tocity, 2] = torch.normal(torch.full((tocity,), meanvelocity), standarddev)
    state[tocity:, 2] = torch.normal(torch.full((outcity,), -meanvelocity), standarddev)

    # x destination
    state[:tocity, 4] = 100+length
    state[tocity:, 4] = -100

    #Y destination
    #state[:n, 5] = width / 2
    #state[n:, 5] = width / 2

    #____________clusters____________________

    #cluster 2

    # Calculate cluster indices
    Ns2 = int(np.floor(tocity * clusters2))  # Number of synchronized individuals
    Ns3 = int(np.floor(tocity * clusters3))
    rp2 = np.random.permutation(tocity) 
    rp3 = np.random.permutation(tocity) # Random shuffle of indices
    cl2 = np.sort(rp2[:Ns2])  # Synchronized indices
    cl3 = np.sort(rp3[:Ns3])

    print(cl2)
    print(cl3)

    # Modify the 0th and 1st columns for rows in iSync
    state[cl2, 0] = state[cl2 - 1, 0]  # Set 0th column to the same as the previous row
    state[cl2, 1] = state[cl2 - 1, 1] + 0.5  # Set 1st column to the previous row's value + 1

    # Modify the 0th and 1st columns for rows in iSync
    state[cl3, 0] = state[cl3 - 2, 0]  # Set 0th column to the same as the previous row
    state[cl3-1, 0] = state[cl3 - 2, 0]  # Set 0th column to the same as the previous row
    state[cl3, 1] = state[cl3 - 2, 1] + 0.5  # Set 1st column to the previous row's value + 1
    state[cl3-1, 1] = state[cl3 - 2, 1] + 1.0  # Set 1st column to the previous row's value + 1

    return state