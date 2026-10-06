import math
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matrix import *
from solver import *
from pedestrian import *

def analyse_pace_synchronisation(
    x_coords,
    y_coords,
    bridge_length,
    dt,
    pace_tolerance=0.05,
    stride_mean=0.75,
    stride_sd=0.08,
    smooth_sec=0.5,
    seed=None,
    velocity_mode="longitudinal"   # "longitudinal" or "2d"
):
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)

    num_timesteps, num_pedestrians = x_coords.shape

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. Velocity from coordinates
    # --------------------------------------------------
    dx = np.diff(x_coords, axis=0)
    dy = np.diff(y_coords, axis=0)

    # Correct periodic boundary jumps only in x-direction
    dx[dx > bridge_length / 2.0] -= bridge_length
    dx[dx < -bridge_length / 2.0] += bridge_length

    vx = dx / dt
    vy = dy / dt

    # --------------------------------------------------
    # 2. Choose velocity definition
    # --------------------------------------------------
    if velocity_mode == "longitudinal":
        # Old/default method:
        # absolute longitudinal velocity only
        speed = np.abs(vx)

    elif velocity_mode == "2d":
        # New method:
        # full 2D walking speed
        speed = np.sqrt(vx**2 + vy**2)

    else:
        raise ValueError("velocity_mode must be 'longitudinal' or '2d'.")

    # --------------------------------------------------
    # 3. Smooth velocity
    # --------------------------------------------------
    window = max(1, int(round(smooth_sec / dt)))
    kernel = np.ones(window) / window

    speed_smooth = np.zeros_like(speed)

    for j in range(num_pedestrians):
        speed_smooth[:, j] = np.convolve(
            speed[:, j],
            kernel,
            mode="same"
        )

    # --------------------------------------------------
    # 4. Assign stride length to each pedestrian
    # --------------------------------------------------
    stride_lengths = rng.normal(
        loc=stride_mean,
        scale=stride_sd,
        size=num_pedestrians
    )

    stride_lengths = np.clip(stride_lengths, 0.45, 1.10)

    # --------------------------------------------------
    # 5. Estimate pace = speed / stride length
    # --------------------------------------------------
    estimated_pace = speed_smooth / stride_lengths[np.newaxis, :]

    # Time vector for velocity and pace
    time_velocity = np.arange(1, num_timesteps) * dt

    # --------------------------------------------------
    # 6. Mean crowd velocity and mean crowd pace at each time
    # --------------------------------------------------
    mean_speed_time = np.mean(speed_smooth, axis=1)
    mean_pace_time = np.mean(estimated_pace, axis=1)

    # --------------------------------------------------
    # 7. Count pedestrians close to instantaneous crowd mean pace
    # --------------------------------------------------
    sync_mask = (
        np.abs(estimated_pace - mean_pace_time[:, np.newaxis])
        <= pace_tolerance
    )

    n_sync_time = np.sum(sync_mask, axis=1)
    frac_sync_time = n_sync_time / num_pedestrians

    return {
        "time_velocity": time_velocity,

        # Components
        "vx": vx,
        "vy": vy,

        # Keep old names so your existing code does not break
        "speed_x": speed,
        "speed_x_smooth": speed_smooth,

        # Also add clearer names
        "speed": speed,
        "speed_smooth": speed_smooth,
        "velocity_mode": velocity_mode,

        "mean_speed_time": mean_speed_time,
        "stride_lengths": stride_lengths,
        "estimated_pace": estimated_pace,
        "mean_pace_time": mean_pace_time,
        "sync_mask": sync_mask,
        "n_sync_time": n_sync_time,
        "frac_sync_time": frac_sync_time,
        "pace_tolerance": pace_tolerance,
    }

# pavesync.py



def compute_pace_sync_summary(
    x_coords,
    y_coords,
    bridge_length,
    dt,
    pace_tolerance=0.05,
    stride_mean=0.75,
    stride_sd=0.08,
    smooth_sec=0.5,
    seed=None,
    print_results=True
):
    """
    Compute overall crowd velocity, pace, and synchronisation statistics.

    Synchronisation definition:
    At each time step, a pedestrian is counted as synchronised if their estimated
    pace is within ±pace_tolerance of the instantaneous crowd mean pace.

    Returns
    -------
    summary : dict
        Dictionary containing scalar crowd statistics.
    """

    # --------------------------------------------------
    # 1. Run existing pace synchronisation analysis
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

    # --------------------------------------------------
    # 2. Extract required quantities
    # --------------------------------------------------
    speed_x_smooth = sync_data["speed_x_smooth"]
    estimated_pace = sync_data["estimated_pace"]

    n_sync_time = sync_data["n_sync_time"]
    frac_sync_time = sync_data["frac_sync_time"]

    num_pedestrians = estimated_pace.shape[1]

    # --------------------------------------------------
    # 3. Overall crowd statistics
    # --------------------------------------------------
    mean_velocity_all = np.nanmean(speed_x_smooth)
    sd_velocity_all = np.nanstd(speed_x_smooth, ddof=1)

    mean_pace_all = np.nanmean(estimated_pace)
    sd_pace_all = np.nanstd(estimated_pace, ddof=1)

    # --------------------------------------------------
    # 4. Synchronisation statistics over time
    # --------------------------------------------------
    mean_n_sync_over_time = np.nanmean(n_sync_time)
    sd_n_sync_over_time = np.nanstd(n_sync_time, ddof=1)

    mean_frac_sync_over_time = np.nanmean(frac_sync_time)
    sd_frac_sync_over_time = np.nanstd(frac_sync_time, ddof=1)

    mean_percent_sync_over_time = 100.0 * mean_frac_sync_over_time

    # --------------------------------------------------
    # 5. Store results
    # --------------------------------------------------
    summary = {
        "mean_abs_velocity": float(mean_velocity_all),
        "sd_abs_velocity": float(sd_velocity_all),

        "overall_mean_pace": float(mean_pace_all),
        "overall_sd_pace": float(sd_pace_all),

        "pace_tolerance": float(pace_tolerance),
        "num_pedestrians": int(num_pedestrians),

        "mean_n_sync_over_time": float(mean_n_sync_over_time),
        "sd_n_sync_over_time": float(sd_n_sync_over_time),

        "mean_frac_sync_over_time": float(mean_frac_sync_over_time),
        "sd_frac_sync_over_time": float(sd_frac_sync_over_time),

        "mean_percent_sync_over_time": float(mean_percent_sync_over_time),
    }

    # --------------------------------------------------
    # 6. Optional print
    # --------------------------------------------------
    if print_results:
        print("\nOverall crowd statistics")
        print("------------------------")
        print(f"Mean absolute velocity                  = {summary['mean_abs_velocity']:.4f} m/s")
        print(f"SD absolute velocity                    = {summary['sd_abs_velocity']:.4f} m/s")
        print(f"Overall mean pace                       = {summary['overall_mean_pace']:.4f} Hz")
        print(f"Overall SD pace                         = {summary['overall_sd_pace']:.4f} Hz")

        print("\nSynchronisation with instantaneous crowd mean pace")
        print("--------------------------------------------------")
        print(f"Tolerance                                = ±{summary['pace_tolerance']:.4f} Hz")
        print(f"Number of pedestrians                    = {summary['num_pedestrians']}")
        print(f"Mean number synchronised over time       = {summary['mean_n_sync_over_time']:.4f}")
        print(f"SD number synchronised over time         = {summary['sd_n_sync_over_time']:.4f}")
        print(f"Mean fraction synchronised over time     = {summary['mean_frac_sync_over_time']:.4f}")
        print(f"SD fraction synchronised over time       = {summary['sd_frac_sync_over_time']:.4f}")
        print(f"Mean percentage synchronised over time   = {summary['mean_percent_sync_over_time']:.2f}%")

    return summary
