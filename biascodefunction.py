import math
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matrix import *
from solver import *
from pedestrian import *


def build_peak_envelope(t, signal):
    """
    Build peak envelope from a single signal using peaks of |signal|.
    """
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)

    abs_sig = np.abs(signal)
    peaks, _ = find_peaks(abs_sig)

    if len(peaks) < 2:
        envelope = abs_sig.copy()
    else:
        x_env = np.concatenate(([t[0]], t[peaks], [t[-1]]))
        y_env = np.concatenate(([abs_sig[0]], abs_sig[peaks], [abs_sig[-1]]))

        f_env = interp1d(
            x_env,
            y_env,
            kind="linear",
            fill_value="extrapolate"
        )
        envelope = f_env(t)

    return envelope


def sliding_rms(signal, dt, window_sec=1.0):
    """
    1-second RMS history (or any window length).
    """
    signal = np.asarray(signal, dtype=float)
    nwin = max(1, int(round(window_sec / dt)))
    kernel = np.ones(nwin, dtype=float) / nwin
    rms = np.sqrt(np.convolve(signal**2, kernel, mode="same"))
    return rms


def percentile_time_history(signals, percentile=95, axis=0):
    """
    Calculate percentile time history from N time histories.

    Parameters
    ----------
    signals : array-like
        Shape should be (N, nt), where:
        N  = number of simulations/time histories
        nt = number of time steps

    percentile : float
        Required percentile, e.g. 95, 90, 50, 5.

    axis : int
        Axis over which to calculate percentile.
        Default axis=0 means calculate percentile across simulations.

    Returns
    -------
    perc_history : ndarray
        Percentile time history with shape (nt,).
    """
    signals = np.asarray(signals, dtype=float)

    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array with shape (N, nt)")

    perc_history = np.percentile(signals, percentile, axis=axis)

    return perc_history




def peak_rms_distribution(accelerations, dt, window_sec=1.0):
    """
    From N acceleration time histories, calculate distribution of peak 1-sec RMS.

    accelerations shape = (N, nt)

    Returns
    -------
    rms_histories : shape (N, nt)
        1-sec RMS time history for each simulation.

    peak_rms_values : shape (N,)
        Maximum 1-sec RMS value from each simulation.
    """
    accelerations = np.asarray(accelerations, dtype=float)

    if accelerations.ndim != 2:
        raise ValueError("accelerations must have shape (N, nt)")

    rms_histories = np.array([
        sliding_rms(acc, dt=dt, window_sec=window_sec)
        for acc in accelerations
    ])

    peak_rms_values = np.max(rms_histories, axis=1)

    return rms_histories, peak_rms_values


def percentile_acc_distribution(accelerations, percentile=95, use_abs=True):
    """
    From N acceleration time histories, calculate distribution of percentile acceleration.

    For each time history:
        a_j(t) -> Q_percentile(|a_j(t)|)

    accelerations shape = (N, nt)

    Returns
    -------
    percentile_values : shape (N,)
        One percentile acceleration value from each simulation.
    """
    accelerations = np.asarray(accelerations, dtype=float)

    if accelerations.ndim != 2:
        raise ValueError("accelerations must have shape (N, nt)")

    if use_abs:
        accelerations = np.abs(accelerations)

    percentile_values = np.percentile(accelerations, percentile, axis=1)

    return percentile_values


def empirical_pdf(samples, bins=30):
    """
    Estimate PDF from scalar samples using histogram.
    """
    samples = np.asarray(samples, dtype=float)

    pdf, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centres, pdf, bin_edges

def kde_pdf(samples, num_points=300, bandwidth=None, positive_only=True):
    """
    Estimate PDF using Gaussian kernel density estimation.

    Parameters
    ----------
    samples : array-like
        Scalar samples, shape = (N,)

    num_points : int
        Number of points used to draw the smooth PDF curve.

    bandwidth : str, float, or None
        KDE bandwidth.
        None or "scott" = default Scott rule.
        "silverman" = Silverman rule.
        float = custom bandwidth factor.

    positive_only : bool
        If True, lower x-limit is clipped at zero.

    Returns
    -------
    x_grid : ndarray
        Points where PDF is evaluated.

    pdf_values : ndarray
        KDE PDF values.
    """

    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for KDE.")

    if np.std(samples) == 0:
        raise ValueError("All samples are identical; KDE cannot be estimated.")

    kde = gaussian_kde(samples, bw_method=bandwidth)

    x_min = np.min(samples)
    x_max = np.max(samples)

    padding = 0.15 * (x_max - x_min)

    if positive_only:
        x_start = max(0.0, x_min - padding)
    else:
        x_start = x_min - padding

    x_end = x_max + padding

    x_grid = np.linspace(x_start, x_end, num_points)
    pdf_values = kde(x_grid)

    return x_grid, pdf_values

def plot_pdf(samples, bins=30, xlabel="Response", title="PDF"):
    """
    Plot empirical PDF from scalar samples.
    """
    bin_centres, pdf, bin_edges = empirical_pdf(samples, bins=bins)

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centres, pdf, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("PDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return bin_centres, pdf, bin_edges

def peak_abs_acc_distribution(accelerations):
    """
    From N acceleration time histories, calculate the distribution
    of peak absolute acceleration.

    Parameters
    ----------
    accelerations : array-like
        Shape = (N, nt), where:
        N  = number of simulations
        nt = number of time steps

    Returns
    -------
    peak_abs_values : ndarray
        Shape = (N,)
        One peak absolute acceleration value from each time history.
    """

    accelerations = np.asarray(accelerations, dtype=float)

    if accelerations.ndim != 2:
        raise ValueError("accelerations must have shape (N, nt)")

    peak_abs_values = np.max(np.abs(accelerations), axis=1)

    return peak_abs_values

def plot_kde_pdf(samples, num_points=300, bandwidth=None,
                 xlabel="Response", title="KDE PDF", positive_only=True):
    """
    Plot KDE-based PDF.
    """

    x_grid, pdf_values = kde_pdf(
        samples,
        num_points=num_points,
        bandwidth=bandwidth,
        positive_only=positive_only
    )

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, pdf_values, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("PDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return x_grid, pdf_values

def run_code_response(
    density,
    mass_ratio,
    length=50.0,
    width=2.0,
    beam_freq=2.0,
    modal_damping_ratio=0.005,
    linear_mass=500.0,
    hht=0.01,
    t_end=100.0,
    numbers=2,
    second_mode_factor=4.0,
    modify_both_modes=True,
    force_reduction_factor=1.0,
):
    """
    Deterministic code-model response with modal mass modified using:

        modifier = 1 + mass_ratio

    Parameters
    ----------
    density : float
        Pedestrian density in ped/m^2.
    mass_ratio : float
        Modal mass ratio from stochastic simulation.
    length, width : float
        Bridge dimensions.
    beam_freq : float or array-like
        If float, use [f1, 4*f1].
        If array/list, use directly.
    modal_damping_ratio : float
        Structural damping ratio.
    linear_mass : float
        Bridge linear mass in kg/m.
    hht : float
        Time step.
    t_end : float
        End time.
    numbers : int
        Number of modes used.
    second_mode_factor : float
        Used only when beam_freq is scalar.
    modify_both_modes : bool
        True  -> modify both modal masses
        False -> modify first mode only

    Returns
    -------
    result : dict
        Contains time, acceleration, envelope, RMS history, etc.
    """

    if numbers not in [1, 2]:
        raise ValueError("This function currently supports numbers = 1 or 2 only.")

    beamFreq = beam_freq

    x_interested = length / 2.0

    # --------------------------------------------------
    # mode shapes
    # --------------------------------------------------
    def curve1(x):
        return np.sin(np.pi * x / length)

    def curve2(x):
        return np.sin(2.0 * np.pi * x / length)

    func_list = [curve1, curve2][:numbers]

    # --------------------------------------------------
    # modal mass modification
    # old: ModalMass = 1.28 * linearMass * length / 2
    # new: ModalMass = (1 + mass_ratio) * linearMass * length / 2
    # --------------------------------------------------
    base_modal_mass = linear_mass * length / 2.0
    modifier = 1.0 + mass_ratio
    modified_modal_mass = modifier * base_modal_mass

    if modify_both_modes:
        modalmass = [modified_modal_mass for _ in range(numbers)]
    else:
        modalmass = [modified_modal_mass] + [base_modal_mass for _ in range(numbers - 1)]

    # --------------------------------------------------
    # bridge stiffness
    # --------------------------------------------------
    modulus = linear_mass * ((2.0 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2

    Bridge = bridge(
        length=length,
        modulus=modulus,
        density=linear_mass,
        damp=modal_damping_ratio,
        numbers=numbers,
        freq=beamFreq,
    )

    # total pedestrians
    num_pedestrians = density * length * width
    deck_area = length * width

    # --------------------------------------------------
    # deterministic code model
    # --------------------------------------------------
    t, u, du, ddu = Newmarksuper_Code(
        Bridge,
        numbers,
        length,
        hht,
        t_end,
        width,
        beamFreq,
        density,
        modal_damping_ratio,
        num_pedestrians,
        deck_area,
        modalmass,
        func_list,
        force_reduction_factor=force_reduction_factor
    )

    accn_hsi = accdyn_super_social(Bridge, ddu, x_interested, modalmass, func_list)
    accn_hsi = np.asarray(accn_hsi, dtype=float)

    # --------------------------------------------------
    # post-processing
    # --------------------------------------------------
    envelope = build_peak_envelope(t, accn_hsi)
    rms_1s = sliding_rms(accn_hsi, hht, window_sec=1.0)

    result = {
        "t": np.asarray(t, dtype=float),
        "acceleration": accn_hsi,
        "envelope": envelope,
        "rms_1s": rms_1s,
        "peak_abs_acceleration": np.max(np.abs(accn_hsi)),
        "peak_envelope": np.max(envelope),
        "max_rms_1s": np.max(rms_1s),
        "modifier": modifier,
        "base_modal_mass": base_modal_mass,
        "modified_modal_mass": modified_modal_mass,
        "modalmass": modalmass,
        "beamFreq": beamFreq,
        "density": density,
        "mass_ratio": mass_ratio,
    }

    return result


def plot_code_response(result):
    """
    Plot time history and peak envelope.
    """
    t = result["t"]
    acc = result["acceleration"]
    env = result["envelope"]

    plt.figure(figsize=(10, 6))
    plt.plot(t, acc, label="code acceleration", linewidth=1.0)
    plt.plot(t, env, label="peak envelope", linewidth=2.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Mid-span acceleration and peak envelope")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_code_response_with_rms(result):
    """
    Plot time history, envelope, and 1-sec RMS.
    """
    t = result["t"]
    acc = result["acceleration"]
    env = result["envelope"]
    rms_1s = result["rms_1s"]

    plt.figure(figsize=(10, 6))
    plt.plot(t, acc, label="code acceleration", linewidth=0.8)
    plt.plot(t, env, label="peak envelope", linewidth=2.0)
    plt.plot(t, rms_1s, label="1-sec RMS", linewidth=2.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Mid-span acceleration, peak envelope, and 1-sec RMS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_bias_from_histories(acc_ref, acc_code, dt):
    """
    Bias based on response histories:
    - peak absolute acceleration bias
    - peak envelope bias
    - max 1-sec RMS bias
    """
    acc_ref = np.asarray(acc_ref, dtype=float)
    acc_code = np.asarray(acc_code, dtype=float)

    n = min(len(acc_ref), len(acc_code))
    acc_ref = acc_ref[:n]
    acc_code = acc_code[:n]

    t = np.arange(n) * dt

    env_ref = build_peak_envelope(t, acc_ref)
    env_code = build_peak_envelope(t, acc_code)

    rms_ref = sliding_rms(acc_ref, dt, window_sec=1.0)
    rms_code = sliding_rms(acc_code, dt, window_sec=1.0)

    eps = 1e-12

    return {
        "bias_peak_abs": np.mean(np.abs(acc_ref)) / max(np.mean(np.abs(acc_code)), eps),
        "bias_peak_envelope": np.mean(env_ref) / max(np.mean(env_code), eps),
        "bias_max_rms_1s": np.mean(rms_ref) / max(np.mean(rms_code), eps),
    }

def run_code_response_from_mc_inputs(
    Tocity,
    Outcity,
    Length,
    Width,
    BeamFreq,
    Beamdamp,
    linear_mass,
    hht,
    t_end,
    mean_mass_ratio,
    mass_ratio_limit=0.05,
    steady_start_time=10.0,
    numbers=1,
    modify_both_modes=False,
    force_reduction_factor=1.0
):
    """
    Run deterministic/code response using the same input condition as MC.

    If mean modal mass ratio <= 5%, no mass ratio modification is applied.
    If mean modal mass ratio > 5%, the MC mean mass ratio is used.

    Returns
    -------
    code_result : dict
        Original result from run_code_response.

    code_measures : dict
        Scalar response measures from deterministic code response.
    """

    density = (Tocity + Outcity) / (Width * Length)

    if mean_mass_ratio > mass_ratio_limit:
        code_mass_ratio = mean_mass_ratio
        mass_ratio_modified = True
    else:
        code_mass_ratio = 0.0
        mass_ratio_modified = False

    code_result = run_code_response(
        density=density,
        mass_ratio=code_mass_ratio,
        length=Length,
        width=Width,
        beam_freq=np.asarray(BeamFreq[:numbers]),
        modal_damping_ratio=np.asarray(Beamdamp),
        linear_mass=linear_mass,
        hht=hht,
        t_end=t_end,
        numbers=numbers,
        modify_both_modes=modify_both_modes,
        force_reduction_factor=force_reduction_factor
    )

    t_code = code_result["t"]
    acc_code = code_result["acceleration"]
    env_code = code_result["envelope"]

    # Discard first 10 seconds
    start_idx = int(round(steady_start_time / hht))

    if start_idx >= len(acc_code):
        raise ValueError("steady_start_time is longer than the code time history.")

    t_code_ss = t_code[start_idx:]
    acc_code_ss = acc_code[start_idx:]
    env_code_ss = env_code[start_idx:]

    # 1-sec RMS
    rms_code_ss = sliding_rms(acc_code_ss, dt=hht, window_sec=1.0)

    # Scalar response measures, same type as MC
    code_measures = {
        "density": density,
        "mean_mass_ratio_from_mc": mean_mass_ratio,
        "code_mass_ratio_used": code_mass_ratio,
        "mass_ratio_modified": mass_ratio_modified,

        "t_ss": t_code_ss,
        "acc_ss": acc_code_ss,
        "env_ss": env_code_ss,
        "rms_1s_ss": rms_code_ss,

        "peak_abs_acc": np.max(np.abs(acc_code_ss)),
        "peak_1sec_rms": np.max(rms_code_ss),
        "p95_abs_acc": np.percentile(np.abs(acc_code_ss), 95),
    }

    return code_result, code_measures


def _average_duplicate_x(x, y):
    """
    Remove duplicate x-values by averaging their y-values.
    This is useful because bridge_frequencies has duplicate values, e.g. 0.5.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    unique_x = np.unique(x)
    unique_y = np.array([
        np.mean(y[x == xx])
        for xx in unique_x
    ])

    sort_idx = np.argsort(unique_x)
    return unique_x[sort_idx], unique_y[sort_idx]


def interpolate_accel_ratio_for_exact_damping(
    results_all,
    beamdamp,
    density,
    target_bf,
    bridge_frequencies,
    pedBodyF=3.10
):
    """
    Interpolate acceleration ratio for one exact damping ratio and density.

    Interpolation is done on x = pedBodyF / bridge_frequency.
    """

    x_vals = pedBodyF / np.asarray(bridge_frequencies, dtype=float)

    y_vals = []
    for bf in bridge_frequencies:
        arr = np.array(results_all[beamdamp][bf][density]["accel_ratio"])
        y_vals.append(np.mean(arr))

    y_vals = np.asarray(y_vals, dtype=float)

    # remove duplicate x values
    x_unique, y_unique = _average_duplicate_x(x_vals, y_vals)

    x_target = pedBodyF / target_bf

    if x_target < x_unique.min() or x_target > x_unique.max():
        raise ValueError(
            f"x_target = {x_target:.4f} is outside interpolation range "
            f"[{x_unique.min():.4f}, {x_unique.max():.4f}]. "
            "Avoid extrapolation unless you really need it."
        )

    y_target = np.interp(x_target, x_unique, y_unique)

    return x_target, y_target


def interpolate_accel_ratio(
    results_all,
    target_beamdamp,
    density,
    target_bf,
    damping_values,
    bridge_frequencies,
    pedBodyF=3.10
):
    """
    Interpolate acceleration ratio for any target bridge frequency and damping ratio.

    First interpolates along frequency-ratio curves for each damping.
    Then interpolates along damping.
    """

    damping_values = np.asarray(damping_values, dtype=float)

    if target_beamdamp < damping_values.min() or target_beamdamp > damping_values.max():
        raise ValueError(
            f"target_beamdamp = {target_beamdamp} is outside damping range "
            f"[{damping_values.min()}, {damping_values.max()}]."
        )

    y_by_damping = []

    for beamdamp in damping_values:
        _, y_val = interpolate_accel_ratio_for_exact_damping(
            results_all=results_all,
            beamdamp=beamdamp,
            density=density,
            target_bf=target_bf,
            bridge_frequencies=bridge_frequencies,
            pedBodyF=pedBodyF
        )

        y_by_damping.append(y_val)

    y_by_damping = np.asarray(y_by_damping)

    y_target = np.interp(target_beamdamp, damping_values, y_by_damping)
    x_target = pedBodyF / target_bf

    return x_target, y_target


def average_duplicate_x(x, *ys):
    """
    Average y-values when duplicate x-values exist.
    Useful because bridge_frequencies may contain duplicate values.
    """
    x = np.asarray(x, dtype=float)
    ys = [np.asarray(y, dtype=float) for y in ys]

    unique_x = np.unique(x)

    averaged_ys = []
    for y in ys:
        averaged_y = np.array([
            np.mean(y[x == xx])
            for xx in unique_x
        ])
        averaged_ys.append(averaged_y)

    sort_idx = np.argsort(unique_x)
    unique_x = unique_x[sort_idx]
    averaged_ys = [y[sort_idx] for y in averaged_ys]

    return (unique_x, *averaged_ys)


def bracket_values(grid, target, name="value"):
    """
    Find lower and upper grid values around target.
    If target exactly exists, lower = upper = target.
    """
    grid = np.asarray(sorted(grid), dtype=float)

    if target < grid.min() or target > grid.max():
        raise ValueError(
            f"{name} = {target} is outside saved range "
            f"[{grid.min()}, {grid.max()}]"
        )

    exact = np.where(np.isclose(grid, target, rtol=0, atol=1e-12))[0]
    if len(exact) > 0:
        val = grid[exact[0]]
        return val, val

    idx_upper = np.searchsorted(grid, target)
    lower = grid[idx_upper - 1]
    upper = grid[idx_upper]

    return lower, upper


def get_curve_for_one_density(
    results_all,
    beamdamp,
    density,
    bridge_frequencies,
    pedBodyF
):
    """
    For one damping and one original density case, extract:

    x = pedBodyF / bridge_frequency
    y = force factor = mean acceleration ratio
    mr = mean modal mass ratio
    """

    x_vals = []
    ff_vals = []
    mr_vals = []

    for bf in bridge_frequencies:
        data = results_all[beamdamp][bf][density]

        accel_ratio_arr = np.asarray(data["accel_ratio"], dtype=float)
        modal_mass_arr = np.asarray(data["modalpedmass_i"], dtype=float)

        x_vals.append(pedBodyF / bf)
        ff_vals.append(np.mean(accel_ratio_arr))
        mr_vals.append(np.mean(modal_mass_arr))

    x_vals = np.asarray(x_vals)
    ff_vals = np.asarray(ff_vals)
    mr_vals = np.asarray(mr_vals)

    # remove duplicate x values
    x_unique, ff_unique, mr_unique = average_duplicate_x(
        x_vals,
        ff_vals,
        mr_vals
    )

    return x_unique, ff_unique, mr_unique


def interpolate_force_factor_one_damping(
    results_all,
    target_bf,
    target_mass_ratio,
    beamdamp,
    densities,
    bridge_frequencies,
    pedBodyF
):
    """
    For one damping value:

    1. Convert target bridge frequency to x = pedBodyF / target_bf.
    2. For each original density curve, interpolate force factor at x.
    3. Also interpolate modal mass ratio at x.
    4. Interpolate force factor across modal mass ratio.
    """

    x_target = pedBodyF / target_bf

    mass_ratio_points = []
    force_factor_points = []

    for density in densities:
        x_curve, ff_curve, mr_curve = get_curve_for_one_density(
            results_all=results_all,
            beamdamp=beamdamp,
            density=density,
            bridge_frequencies=bridge_frequencies,
            pedBodyF=pedBodyF
        )

        if x_target < x_curve.min() or x_target > x_curve.max():
            raise ValueError(
                f"x_target = {x_target:.4f} is outside saved x range "
                f"[{x_curve.min():.4f}, {x_curve.max():.4f}]"
            )

        ff_at_x = np.interp(x_target, x_curve, ff_curve)
        mr_at_x = np.interp(x_target, x_curve, mr_curve)

        force_factor_points.append(ff_at_x)
        mass_ratio_points.append(mr_at_x)

    mass_ratio_points = np.asarray(mass_ratio_points)
    force_factor_points = np.asarray(force_factor_points)

    # sort by mass ratio
    sort_idx = np.argsort(mass_ratio_points)
    mr_sorted = mass_ratio_points[sort_idx]
    ff_sorted = force_factor_points[sort_idx]

    # remove duplicate mass-ratio points if any
    mr_unique, ff_unique = average_duplicate_x(mr_sorted, ff_sorted)

    if target_mass_ratio < mr_unique.min() or target_mass_ratio > mr_unique.max():
        raise ValueError(
            f"target_mass_ratio = {target_mass_ratio:.4f} is outside saved "
            f"mass-ratio range [{mr_unique.min():.4f}, {mr_unique.max():.4f}] "
            f"at damping = {beamdamp} and bridge frequency = {target_bf} Hz."
        )

    ff_at_mass_ratio = np.interp(
        target_mass_ratio,
        mr_unique,
        ff_unique
    )

    return x_target, ff_at_mass_ratio, mr_unique, ff_unique


def interpolate_force_factor_exact_point(
    results_all,
    target_bf,
    target_beamdamp,
    target_mass_ratio,
    damping_values,
    densities,
    bridge_frequencies,
    pedBodyF
):
    """
    Interpolate exact force factor at:

        target bridge frequency
        target damping ratio
        target modal mass ratio

    Final output:
        x_target = pedBodyF / target_bf
        force_factor_target
    """

    lower_damp, upper_damp = bracket_values(
        damping_values,
        target_beamdamp,
        name="target_beamdamp"
    )

    # If damping exactly exists in saved results
    if np.isclose(lower_damp, upper_damp):
        x_target, ff_target, mr_grid, ff_grid = interpolate_force_factor_one_damping(
            results_all=results_all,
            target_bf=target_bf,
            target_mass_ratio=target_mass_ratio,
            beamdamp=lower_damp,
            densities=densities,
            bridge_frequencies=bridge_frequencies,
            pedBodyF=pedBodyF
        )

        details = {
            "lower_damp": lower_damp,
            "upper_damp": upper_damp,
            "ff_lower": ff_target,
            "ff_upper": ff_target,
            "damping_weight": 0.0,
            "mr_grid_lower": mr_grid,
            "ff_grid_lower": ff_grid,
            "mr_grid_upper": mr_grid,
            "ff_grid_upper": ff_grid,
        }

        return x_target, ff_target, details

    # Interpolate at lower damping
    x_target, ff_lower, mr_grid_lower, ff_grid_lower = interpolate_force_factor_one_damping(
        results_all=results_all,
        target_bf=target_bf,
        target_mass_ratio=target_mass_ratio,
        beamdamp=lower_damp,
        densities=densities,
        bridge_frequencies=bridge_frequencies,
        pedBodyF=pedBodyF
    )

    # Interpolate at upper damping
    _, ff_upper, mr_grid_upper, ff_grid_upper = interpolate_force_factor_one_damping(
        results_all=results_all,
        target_bf=target_bf,
        target_mass_ratio=target_mass_ratio,
        beamdamp=upper_damp,
        densities=densities,
        bridge_frequencies=bridge_frequencies,
        pedBodyF=pedBodyF
    )

    # Linear interpolation between damping curves
    w = (target_beamdamp - lower_damp) / (upper_damp - lower_damp)

    ff_target = (1.0 - w) * ff_lower + w * ff_upper

    details = {
        "lower_damp": lower_damp,
        "upper_damp": upper_damp,
        "ff_lower": ff_lower,
        "ff_upper": ff_upper,
        "damping_weight": w,
        "mr_grid_lower": mr_grid_lower,
        "ff_grid_lower": ff_grid_lower,
        "mr_grid_upper": mr_grid_upper,
        "ff_grid_upper": ff_grid_upper,
    }

    return x_target, ff_target, details

# ==================================================
# BANDED MODIFICATION FACTOR INTERPOLATION HELPERS
# for new database:
# results_all[band_name][beamdamp][bf][density][factor_key]
# ==================================================

def bracket_values_banded(values, target, name="target"):
    """
    Find lower and upper values around target.
    If target exactly exists, lower = upper = target.
    """
    values = np.array(sorted(values), dtype=float)

    if target < values[0] or target > values[-1]:
        raise ValueError(
            f"{name} = {target} is outside available range "
            f"[{values[0]}, {values[-1]}]"
        )

    exact = np.where(np.isclose(values, target, rtol=0, atol=1e-12))[0]

    if len(exact) > 0:
        val = values[exact[0]]
        return val, val

    idx_upper = np.searchsorted(values, target)

    lower = values[idx_upper - 1]
    upper = values[idx_upper]

    return lower, upper


def interpolate_modification_factor_one_damping_banded(
    results_all,
    bands_to_combine,
    target_bf,
    target_mass_ratio,
    beamdamp,
    densities,
    factor_key="modification_factor"
):
    """
    Interpolate modification factor at one structural damping ratio.

    New database structure:

        results_all[band_name][beamdamp][bf][density][factor_key]

    Steps:
    1. For each density/mass-ratio curve, interpolate over bridge frequency.
    2. Then interpolate over modal mass ratio.
    """

    mass_ratio_grid = []
    factor_grid = []

    for density in densities:

        freq_to_values = {}
        mass_ratio_values = []

        for band_name in bands_to_combine:

            if band_name not in results_all:
                continue

            bridge_frequencies_band = sorted(
                results_all[band_name][beamdamp].keys()
            )

            for bf in bridge_frequencies_band:

                data = results_all[band_name][beamdamp][bf][density]

                factor_arr = np.asarray(data[factor_key], dtype=float)
                mr_arr = np.asarray(data["modalpedmass_i"], dtype=float)

                factor_mean = np.mean(factor_arr)
                mr_mean = np.mean(mr_arr)

                if bf not in freq_to_values:
                    freq_to_values[bf] = []

                freq_to_values[bf].append(factor_mean)
                mass_ratio_values.append(mr_mean)

        freqs = np.array(sorted(freq_to_values.keys()), dtype=float)

        factors = np.array(
            [np.mean(freq_to_values[bf]) for bf in freqs],
            dtype=float
        )

        if target_bf < freqs.min() or target_bf > freqs.max():
            raise ValueError(
                f"target_bf = {target_bf} is outside available frequency range "
                f"[{freqs.min()}, {freqs.max()}]"
            )

        # interpolate over bridge frequency for this density
        factor_at_bf = np.interp(target_bf, freqs, factors)

        # representative modal mass ratio for this density curve
        mass_ratio_mean = np.mean(mass_ratio_values)

        mass_ratio_grid.append(mass_ratio_mean)
        factor_grid.append(factor_at_bf)

    mass_ratio_grid = np.array(mass_ratio_grid, dtype=float)
    factor_grid = np.array(factor_grid, dtype=float)

    # sort by mass ratio
    sort_idx = np.argsort(mass_ratio_grid)
    mass_ratio_grid = mass_ratio_grid[sort_idx]
    factor_grid = factor_grid[sort_idx]

    if target_mass_ratio < mass_ratio_grid.min() or target_mass_ratio > mass_ratio_grid.max():
        raise ValueError(
            f"target_mass_ratio = {target_mass_ratio} is outside available range "
            f"[{mass_ratio_grid.min()}, {mass_ratio_grid.max()}]"
        )

    # interpolate over modal mass ratio
    factor_target = np.interp(
        target_mass_ratio,
        mass_ratio_grid,
        factor_grid
    )

    return factor_target, mass_ratio_grid, factor_grid


def interpolate_modification_factor_banded(
    results_all,
    target_bf,
    target_beamdamp,
    target_mass_ratio,
    damping_values,
    densities,
    bands_to_combine=None,
    factor_key="modification_factor"
):
    """
    Interpolate modification factor at:

        target bridge frequency
        target structural damping ratio
        target modal mass ratio

    For new database structure:

        results_all[band_name][beamdamp][bf][density][factor_key]
    """

    if bands_to_combine is None:
        bands_to_combine = list(results_all.keys())

    lower_damp, upper_damp = bracket_values_banded(
        damping_values,
        target_beamdamp,
        name="target_beamdamp"
    )

    # If damping exactly exists
    if np.isclose(lower_damp, upper_damp):

        ff_target, mr_grid, ff_grid = interpolate_modification_factor_one_damping_banded(
            results_all=results_all,
            bands_to_combine=bands_to_combine,
            target_bf=target_bf,
            target_mass_ratio=target_mass_ratio,
            beamdamp=lower_damp,
            densities=densities,
            factor_key=factor_key
        )

        details = {
            "lower_damp": lower_damp,
            "upper_damp": upper_damp,
            "ff_lower": ff_target,
            "ff_upper": ff_target,
            "damping_weight": 0.0,
            "mr_grid_lower": mr_grid,
            "ff_grid_lower": ff_grid,
            "mr_grid_upper": mr_grid,
            "ff_grid_upper": ff_grid,
        }

        return ff_target, details

    # Interpolate at lower damping
    ff_lower, mr_grid_lower, ff_grid_lower = interpolate_modification_factor_one_damping_banded(
        results_all=results_all,
        bands_to_combine=bands_to_combine,
        target_bf=target_bf,
        target_mass_ratio=target_mass_ratio,
        beamdamp=lower_damp,
        densities=densities,
        factor_key=factor_key
    )

    # Interpolate at upper damping
    ff_upper, mr_grid_upper, ff_grid_upper = interpolate_modification_factor_one_damping_banded(
        results_all=results_all,
        bands_to_combine=bands_to_combine,
        target_bf=target_bf,
        target_mass_ratio=target_mass_ratio,
        beamdamp=upper_damp,
        densities=densities,
        factor_key=factor_key
    )

    # interpolate over damping ratio
    w = (target_beamdamp - lower_damp) / (upper_damp - lower_damp)

    ff_target = (1.0 - w) * ff_lower + w * ff_upper

    details = {
        "lower_damp": lower_damp,
        "upper_damp": upper_damp,
        "ff_lower": ff_lower,
        "ff_upper": ff_upper,
        "damping_weight": w,
        "mr_grid_lower": mr_grid_lower,
        "ff_grid_lower": ff_grid_lower,
        "mr_grid_upper": mr_grid_upper,
        "ff_grid_upper": ff_grid_upper,
    }

    return ff_target, details

import numpy as np


def running_mean_sd_1d(values):
    """
    Running mean and sample SD for one scalar value per simulation.

    Example uses:
    - peak 1-sec RMS per simulation
    - peak absolute acceleration per simulation
    - 95% absolute acceleration per simulation
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    sim_numbers = np.arange(1, n + 1)

    if n == 0:
        return sim_numbers, np.array([]), np.array([])

    cum_sum = np.cumsum(values)
    cum_sumsq = np.cumsum(values**2)

    running_mean = cum_sum / sim_numbers

    running_var = np.full(n, np.nan)

    if n > 1:
        running_var[1:] = (
            cum_sumsq[1:] - (cum_sum[1:] ** 2) / sim_numbers[1:]
        ) / (sim_numbers[1:] - 1)

        running_var = np.maximum(running_var, 0.0)

    running_sd = np.sqrt(running_var)

    return sim_numbers, running_mean, running_sd


def running_instantaneous_acc_stats(accelerations_ss):
    """
    Running mean and sample SD for instantaneous acceleration.

    For simulation number n, this combines all acceleration time-history
    values from simulations 1 to n into one large vector.
    """

    accelerations_ss = np.asarray(accelerations_ss, dtype=float)

    if accelerations_ss.ndim != 2:
        raise ValueError(
            "accelerations_ss must have shape (num_simulations, num_time_steps)."
        )

    num_sims, num_time_steps = accelerations_ss.shape
    sim_numbers = np.arange(1, num_sims + 1)

    if num_sims == 0 or num_time_steps == 0:
        return sim_numbers, np.array([]), np.array([])

    sum_per_sim = np.sum(accelerations_ss, axis=1)
    sumsq_per_sim = np.sum(accelerations_ss**2, axis=1)

    cum_sum = np.cumsum(sum_per_sim)
    cum_sumsq = np.cumsum(sumsq_per_sim)

    cum_count = sim_numbers * num_time_steps

    running_mean = cum_sum / cum_count

    running_var = (
        cum_sumsq - (cum_sum**2) / cum_count
    ) / (cum_count - 1)

    running_var = np.maximum(running_var, 0.0)
    running_sd = np.sqrt(running_var)

    return sim_numbers, running_mean, running_sd