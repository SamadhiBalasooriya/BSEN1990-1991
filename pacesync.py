import numpy as np
import torch
import socialforce
from socialforcefunctions import initial_state_corridor
from pacesyncedhelper import analyse_pace_synchronisation
# If your file is actually named pacesyncedhelper.py, use:
# from pacesyncedhelper import analyse_pace_synchronisation


def run_crowd_pace_sync_simulation(args):
    """
    Generate one social-force crowd simulation and compute crowd velocity,
    pace, and synchronisation summary.

    Designed for multiprocessing with pool.map().

    Parameters
    ----------
    args : tuple
        (
            seed,
            Tocity,
            Outcity,
            Length,
            Width,
            meanvelocity,
            stddev,
            hht,
            ped_model,
            pace_tolerance,
            stride_mean,
            stride_sd,
            smooth_sec,
            return_time_histories
        )

    Returns
    -------
    result : dict
        Scalar summary for one simulation.
        If return_time_histories=True, also returns time histories needed for plots.
    """

    (
        seed,
        Tocity,
        Outcity,
        Length,
        Width,
        meanvelocity,
        stddev,
        hht,
        ped_model,
        pace_tolerance,
        stride_mean,
        stride_sd,
        smooth_sec,
        return_time_histories
    ) = args

    # --------------------------------------------------
    # Optional seed for multiprocessing
    # --------------------------------------------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # --------------------------------------------------
    # Basic settings
    # --------------------------------------------------
    length = Length
    width = Width
    dt = hht
    bridge_length = length

    time = np.arange(0, (length + 88) / meanvelocity, hht)
    totalTimeSteps = np.size(time)

    # --------------------------------------------------
    # Initial pedestrian states
    # --------------------------------------------------
    initial_state = initial_state_corridor(
        Tocity,
        Outcity,
        length,
        width,
        meanvelocity,
        stddev
    )

    # --------------------------------------------------
    # Walls / pedestrian space
    # --------------------------------------------------
    upper_wall = torch.stack(
        [
            torch.linspace(0, length, 1000),
            torch.full((1000,), width)
        ],
        -1
    )

    lower_wall = torch.stack(
        [
            torch.linspace(0, length, 1000),
            torch.full((1000,), 0)
        ],
        -1
    )

    ped_space = socialforce.potentials.PedSpacePotential(
        [upper_wall, lower_wall]
    )

    # --------------------------------------------------
    # Pedestrian-pedestrian interaction model
    # --------------------------------------------------
    if ped_model == "2D":
        ped_ped = socialforce.potentials.PedPedPotential2D()

    elif ped_model == "Diamond":
        ped_ped = socialforce.potentials.PedPedPotentialDiamond(
            sigma=0.5
        )

    elif ped_model == "Diamond_asym":
        ped_ped = socialforce.potentials.PedPedPotentialDiamond(
            sigma=0.5,
            asymmetry_angle=-20.0
        )

    else:
        raise ValueError(
            "ped_model must be one of: '2D', 'Diamond', 'Diamond_asym'"
        )

    # --------------------------------------------------
    # Social-force simulator
    # --------------------------------------------------
    simulator = socialforce.Simulator(
        ped_ped=ped_ped,
        ped_space=ped_space,
        oversampling=1,
        delta_t=hht
    )

    simulator.integrator = socialforce.simulator.PeriodicBoundary(
        simulator.integrator,
        x_boundary=[0, +length]
    )

    with torch.no_grad():
        states_sf = simulator.run(initial_state, totalTimeSteps)

    # --------------------------------------------------
    # Extract x/y coordinates
    # --------------------------------------------------
    num_timesteps = len(states_sf)
    num_pedestrians = states_sf[0].shape[0]

    x_coords = np.zeros((num_timesteps, num_pedestrians))
    y_coords = np.zeros((num_timesteps, num_pedestrians))

    for k in range(num_timesteps):
        x_coords[k, :] = states_sf[k][:, 0].numpy()
        y_coords[k, :] = states_sf[k][:, 1].numpy()

    # --------------------------------------------------
    # Velocity, pace, and synchronisation analysis
    # --------------------------------------------------
    sync_data = analyse_pace_synchronisation(
        x_coords=x_coords,
        y_coords=y_coords,
        bridge_length=bridge_length,
        dt=dt,
        pace_tolerance=pace_tolerance,
        stride_mean=stride_mean,
        stride_sd=stride_sd,
        smooth_sec=smooth_sec,
        seed=seed
    )

    speed_x_smooth = sync_data["speed_x_smooth"]
    estimated_pace = sync_data["estimated_pace"]

    n_sync_time = sync_data["n_sync_time"]
    frac_sync_time = sync_data["frac_sync_time"]

    # --------------------------------------------------
    # Overall velocity and pace statistics
    # --------------------------------------------------
    mean_abs_velocity = np.nanmean(speed_x_smooth)
    sd_abs_velocity = np.nanstd(speed_x_smooth, ddof=1)

    overall_mean_pace = np.nanmean(estimated_pace)
    overall_sd_pace = np.nanstd(estimated_pace, ddof=1)

    # --------------------------------------------------
    # Synchronisation statistics
    # n_sync_time is a time history:
    # at each time step, how many pedestrians are close to
    # the instantaneous crowd mean pace.
    # --------------------------------------------------
    mean_n_sync_over_time = np.nanmean(n_sync_time)
    sd_n_sync_over_time = np.nanstd(n_sync_time, ddof=1)

    mean_frac_sync_over_time = np.nanmean(frac_sync_time)
    sd_frac_sync_over_time = np.nanstd(frac_sync_time, ddof=1)

    mean_percent_sync_over_time = 100.0 * mean_frac_sync_over_time

    # --------------------------------------------------
    # Return scalar summary
    # --------------------------------------------------
    result = {
        "seed": seed,
        "Tocity": int(Tocity),
        "Outcity": int(Outcity),
        "num_pedestrians": int(num_pedestrians),
        "num_timesteps": int(num_timesteps),
        "totalTimeSteps": int(totalTimeSteps),
        "Length": float(Length),
        "Width": float(Width),
        "meanvelocity_input": float(meanvelocity),
        "stddev_input": float(stddev),
        "hht": float(hht),
        "ped_model": ped_model,

        "pace_tolerance": float(pace_tolerance),
        "stride_mean": float(stride_mean),
        "stride_sd": float(stride_sd),
        "smooth_sec": float(smooth_sec),

        "mean_abs_velocity": float(mean_abs_velocity),
        "sd_abs_velocity": float(sd_abs_velocity),

        "overall_mean_pace": float(overall_mean_pace),
        "overall_sd_pace": float(overall_sd_pace),

        "mean_n_sync_over_time": float(mean_n_sync_over_time),
        "sd_n_sync_over_time": float(sd_n_sync_over_time),

        "mean_frac_sync_over_time": float(mean_frac_sync_over_time),
        "sd_frac_sync_over_time": float(sd_frac_sync_over_time),

        "mean_percent_sync_over_time": float(mean_percent_sync_over_time),
    }

    # --------------------------------------------------
    # Optional: return time histories for plotting/debugging
    # Do not use this for large MC unless you really need it.
    # --------------------------------------------------
    if return_time_histories:
        result["time_velocity"] = sync_data["time_velocity"]
        result["speed_x_smooth"] = sync_data["speed_x_smooth"]
        result["mean_speed_time"] = sync_data["mean_speed_time"]
        result["estimated_pace"] = sync_data["estimated_pace"]
        result["mean_pace_time"] = sync_data["mean_pace_time"]
        result["n_sync_time"] = sync_data["n_sync_time"]
        result["frac_sync_time"] = sync_data["frac_sync_time"]
        result["x_coords"] = x_coords
        result["y_coords"] = y_coords

    return result