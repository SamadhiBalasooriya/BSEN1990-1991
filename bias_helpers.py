from IPython.display import HTML, display
import numpy as np
#import torch
#from socialforcefunctions import initial_state_corridor,bounded_gamma_sample1
#import pandas as pd
from matrix import*  
from solver import *
from pedestrian import* 
#from matplotlib import pyplot as plt
from bias import run_simulation
from biascodefunction import run_code_response, sliding_rms

def evaluate_bias_for_frequency(fb, num_simulations, num_processes,
                                peddamp, pedBodyF, Tocity, Outcity,
                                Length, Width, hht,
                                density_case, modal_damping_ratio,
                                linear_mass, numbers, modify_both_modes):

    rng = np.random.default_rng()
    seeds = rng.integers(0, 2**31 - 1, size=num_simulations)

    # pass fb into stochastic model
    args_list = [
        (int(seed), peddamp, pedBodyF, Tocity, Outcity, Length, Width, fb)
        for seed in seeds
    ]

    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        results = pool.map(run_simulation, args_list)

    accelerations = np.array([r[0] for r in results], dtype=float)
    mean_mass_ratios = np.array([r[1] for r in results], dtype=float)

    nt = accelerations.shape[1]
    t = np.arange(nt) * hht

    # stochastic 95% curves
    abs_acc_95_stoch = np.percentile(np.abs(accelerations), 95, axis=0)

    rms_all = np.array([
        sliding_rms(accelerations[i], hht, window_sec=1.0)
        for i in range(num_simulations)
    ])
    rms_95_stoch = np.percentile(rms_all, 95, axis=0)

    # scalar summaries
    mean_abs_acc_95_stoch = np.mean(abs_acc_95_stoch)
    mean_rms_95_stoch = np.mean(rms_95_stoch)
    mean_mass_ratio_case = np.mean(mean_mass_ratios)

    # code model for same frequency
    # if you are using 2 modes, you may define second mode as 4*fb
    beam_freq = np.array([fb, 4.0 * fb])

    result_code = run_code_response(
        density=density_case,
        mass_ratio=mean_mass_ratio_case,
        length=Length,
        width=Width,
        beam_freq=beam_freq,
        modal_damping_ratio=modal_damping_ratio,
        linear_mass=linear_mass,
        hht=hht,
        t_end=t[-1] + hht,
        numbers=numbers,
        modify_both_modes=modify_both_modes
    )

    acc_code = np.asarray(result_code["acceleration"], dtype=float)
    nmin = min(len(acc_code), len(t))
    acc_code = acc_code[:nmin]

    rms_code = sliding_rms(acc_code, hht, window_sec=1.0)

    code_abs_acc_95 = np.percentile(np.abs(acc_code), 95)
    code_rms_95 = np.percentile(rms_code, 95)

    eps = 1e-12
    bias_abs = mean_abs_acc_95_stoch / max(code_abs_acc_95, eps)
    bias_rms = mean_rms_95_stoch / max(code_rms_95, eps)

    return {
        "bridge_frequency": fb,
        "mean_mass_ratio": mean_mass_ratio_case,
        "stoch_mean_95_abs": mean_abs_acc_95_stoch,
        "stoch_mean_95_rms": mean_rms_95_stoch,
        "code_95_abs": code_abs_acc_95,
        "code_95_rms": code_rms_95,
        "bias_abs": bias_abs,
        "bias_rms": bias_rms
    }
