from matrix import*  
from solver import *
from pedestrian import* 
from matplotlib import pyplot as plt
import math
import numpy as np

from trialUDL import Bridge, ModalMass, curve1, curve2

from trialUDL import curve2


from matrix import *
from solver import *
from pedestrian import *
import math
import numpy as np


def run_deterministic_response(bridge_frequency, density, length=50, width=2):
    """
    Deterministic/code-model acceleration time history at midspan.

    Parameters
    ----------
    bridge_frequency : float
        First bridge frequency in Hz.
    density : float
        Pedestrian density in ped/m^2.
    length : float, optional
        Bridge length in m.
    width : float, optional
        Bridge width in m.

    Returns
    -------
    t : ndarray
        Time vector.
    accn_hsi : ndarray
        Deterministic midspan acceleration time history.
    """

    # --------------------------------------------------
    # Bridge and model settings
    # --------------------------------------------------
    height = 0.6
    E = 200e9
    modalDampingRatio = 0.005
    beamFreq = np.array([bridge_frequency, 4.0 * bridge_frequency], dtype=float)
    linearMass = 500.0
    x_interested = length / 2
    numbers = 2
    hht = 0.01
    t_end = 100.0

    # --------------------------------------------------
    # Mode shapes and modal properties
    # --------------------------------------------------
    ModalMass = 1.28 * linearMass * length / 2

    def curve1(x):
        return np.sin(np.pi * x / length)

    def curve2(x):
        return np.sin(2 * np.pi * x / length)

    func_list = [curve1, curve2]
    modalmass = [ModalMass, ModalMass]

    # bridge stiffness / modulus
    modulus = linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2

    Bridge = bridge(
        length=length,
        modulus=modulus,
        density=linearMass,
        damp=modalDampingRatio,
        numbers=numbers,
        freq=beamFreq,
    )

    # --------------------------------------------------
    # Deterministic code-model response
    # --------------------------------------------------
    t, u, du, ddu = Newmarksuper_Code(
        Bridge,
        numbers,
        length,
        hht,
        t_end,
        width,
        bridge_frequency,          # walking frequency fs
        density,                   # density ped/m^2
        modalDampingRatio,
        density * length * width,  # number of pedestrians
        length * width,            # deck area S
        modalmass,
        func_list
    )

    accn_hsi = accdyn_super_social(Bridge, ddu, x_interested, modalmass, func_list)

    return t, accn_hsi

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline


# =========================================================
# 1. Load results and compute factors_all
# =========================================================
def compute_factors_from_pickle(
    pkl_path,
    densities=None,
    bridge_frequencies=None,
    use_abs=False,
    eps=1e-12
):
    """
    Load results_all from a pickle file and compute factors_all.

    Expected structure:
        results_all[bridge_frequency][density] = {
            "modalpedmass_i": ...,
            "eq_UDL": ...,
            "realUDL": ...
        }

    Parameters
    ----------
    pkl_path : str
        Path to force_factors_final2.pkl
    densities : list or None
        Densities to use. If None, inferred from pickle.
    bridge_frequencies : list or None
        Frequencies to use. If None, inferred from pickle.
    use_abs : bool
        If True, compute percentiles from absolute time histories.
        If False, follows your original code exactly.
    eps : float
        Small number to avoid divide-by-zero in pointwise ratios.

    Returns
    -------
    factors_all : list of dict
    results_all : dict
    """
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    # unwrap the saved structure
    if "results_all" in loaded:
        results_all = loaded["results_all"]
        settings = loaded.get("settings", {})
    else:
        results_all = loaded
        settings = {}

    if bridge_frequencies is None:
        bridge_frequencies = settings.get("bridge_frequencies", sorted(results_all.keys()))

    if densities is None:
        densities = settings.get("densities", sorted(results_all[bridge_frequencies[0]].keys()))

    factors_all = []

    for bf in bridge_frequencies:
        for density in densities:
            if bf not in results_all:
                print(f"Skipping bf={bf}: not found in pickle")
                continue
            if density not in results_all[bf]:
                print(f"Skipping bf={bf}, density={density}: not found in pickle")
                continue

            data = results_all[bf][density]

            modalpedmass_i = np.array(data["modalpedmass_i"], dtype=float)
            eq_UDL = np.array(data["eq_UDL"], dtype=float)
            realUDL = np.array(data["realUDL"], dtype=float)

            # percentile time histories
            if use_abs:
                w_eq_95 = np.percentile(np.abs(eq_UDL), 95, axis=0)
                w_real_95 = np.percentile(np.abs(realUDL), 95, axis=0)
            else:
                w_eq_95 = np.percentile(eq_UDL, 95, axis=0)
                w_real_95 = np.percentile(realUDL, 95, axis=0)

            denom_ls = np.dot(w_eq_95, w_eq_95)
            factor_ls = np.dot(w_eq_95, w_real_95) / denom_ls if abs(denom_ls) > eps else np.nan

            max_eq = np.max(w_eq_95)
            rms_eq = np.sqrt(np.mean(w_eq_95**2))

            factor_peak = np.max(w_real_95) / max_eq if abs(max_eq) > eps else np.nan
            factor_rms = np.sqrt(np.mean(w_real_95**2)) / rms_eq if abs(rms_eq) > eps else np.nan

            valid = np.abs(w_eq_95) > eps
            ratio_pointwise = w_real_95[valid] / w_eq_95[valid]

            factor_median_ratio = np.median(ratio_pointwise) if ratio_pointwise.size else np.nan
            factor_mean_ratio = np.mean(ratio_pointwise) if ratio_pointwise.size else np.nan

            mass_ratio = np.mean(modalpedmass_i)

            factors_all.append({
                "bridge_frequency": float(bf),
                "density": float(density),
                "mass_ratio": float(mass_ratio),
                "factor_ls": float(factor_ls),
                "factor_peak": float(factor_peak),
                "factor_rms": float(factor_rms),
                "factor_median_ratio": float(factor_median_ratio),
                "factor_mean_ratio": float(factor_mean_ratio),
            })

            print(
                f"bf={bf:.3f}, density={density:.4f}, "
                f"mass_ratio={mass_ratio:.6f}, RMS={factor_rms:.4f}"
            )

    return factors_all, results_all


# =========================================================
# 2. Build interpolation model
# =========================================================
def build_force_factor_interpolator(
    factors_all,
    factor_key="factor_rms",
    invert=True,
    mass_ratio_round=3
):
    """
    Build interpolation curves from factors_all.

    Parameters
    ----------
    factors_all : list of dict
    factor_key : str
        Which factor to use, e.g. "factor_rms", "factor_peak", "factor_ls"
    invert : bool
        If True, y = 1 / factor
        If False, y = factor
    mass_ratio_round : int
        Group mass ratios by rounding to this many decimals

    Returns
    -------
    model : dict
        Contains arrays, grouped curves, and query helpers
    """
    bridge_freq_arr = np.array([d["bridge_frequency"] for d in factors_all], dtype=float)
    density_arr     = np.array([d["density"] for d in factors_all], dtype=float)
    mass_ratio_arr  = np.array([d["mass_ratio"] for d in factors_all], dtype=float)
    factor_arr      = np.array([d[factor_key] for d in factors_all], dtype=float)

    if invert:
        y_all = 1.0 / factor_arr
    else:
        y_all = factor_arr

    mass_ratio_group = np.round(mass_ratio_arr, mass_ratio_round)
    unique_mass_ratios = np.unique(mass_ratio_group)

    curves = {}

    for mval in unique_mass_ratios:
        idx = np.where(mass_ratio_group == mval)[0]

        x = 3.1 / bridge_freq_arr[idx]
        y = y_all[idx]

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, inverse = np.unique(x, return_inverse=True)
        y_unique = np.array([y[inverse == i].mean() for i in range(len(x_unique))])

        curves[mval] = (x_unique, y_unique)

    def interpolate_on_curve(x_target, x_data, y_data):
        if len(x_data) == 1:
            return float(y_data[0])
        f = PchipInterpolator(x_data, y_data, extrapolate=True)
        return float(f(x_target))

    def query(bridge_frequency, mass_ratio):
        x_target = 3.1 / bridge_frequency
        mvals = np.array(sorted(curves.keys()), dtype=float)

        if len(mvals) < 2:
            raise ValueError("Need at least two mass-ratio groups to interpolate.")

        # exact group available
        if np.any(np.isclose(mass_ratio, mvals, atol=1e-12)):
            m_exact = mvals[np.argmin(np.abs(mvals - mass_ratio))]
            x_data, y_data = curves[m_exact]
            y_target = interpolate_on_curve(x_target, x_data, y_data)
            return x_target, y_target

        # bracket in mass ratio
        if mass_ratio <= mvals[0]:
            m1, m2 = mvals[0], mvals[1]
        elif mass_ratio >= mvals[-1]:
            m1, m2 = mvals[-2], mvals[-1]
        else:
            idx_upper = np.searchsorted(mvals, mass_ratio)
            m1, m2 = mvals[idx_upper - 1], mvals[idx_upper]

        x1, y1_data = curves[m1]
        x2, y2_data = curves[m2]

        y1 = interpolate_on_curve(x_target, x1, y1_data)
        y2 = interpolate_on_curve(x_target, x2, y2_data)

        y_target = y1 + (y2 - y1) * (mass_ratio - m1) / (m2 - m1)
        return x_target, float(y_target)

    model = {
        "bridge_freq_arr": bridge_freq_arr,
        "density_arr": density_arr,
        "mass_ratio_arr": mass_ratio_arr,
        "factor_arr": factor_arr,
        "y_all": y_all,
        "mass_ratio_group": mass_ratio_group,
        "unique_mass_ratios": unique_mass_ratios,
        "curves": curves,
        "factor_key": factor_key,
        "invert": invert,
        "query": query,
    }

    return model


# =========================================================
# 3. Plot helper
# =========================================================
def plot_force_factor_curves(
    model,
    bridge_frequency_input=None,
    mass_ratio_input=None,
    xlim=(0, 8.0),
    ylim=None
):
    """
    Plot grouped curves and optionally a queried point.
    """
    curves = model["curves"]
    unique_mass_ratios = np.array(sorted(curves.keys()), dtype=float)

    plt.figure(figsize=(10, 6))

    for mval in unique_mass_ratios:
        x_unique, y_unique = curves[mval]

        if len(x_unique) > 5:
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), 1500)
            spline = UnivariateSpline(x_unique, y_unique, s=0.001)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, linewidth=2, label=f"mass ratio = {mval:.3f}")
        else:
            plt.plot(x_unique, y_unique, "-o", linewidth=2, label=f"mass ratio = {mval:.3f}")

    # optional query point
    if bridge_frequency_input is not None and mass_ratio_input is not None:
        x_query, y_query = model["query"](bridge_frequency_input, mass_ratio_input)

        plt.scatter(
            x_query, y_query,
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            zorder=10,
            label=f"query point ({bridge_frequency_input:.2f} Hz, {mass_ratio_input:.4f})"
        )

        plt.text(
            x_query, y_query + 0.02,
            f"y = {y_query:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    bridge_freq_marks = np.array([1.0, 1.7, 2.1, 2.6, 5.0], dtype=float)
    x_marks = 3.1 / bridge_freq_marks

    for bf, xm in zip(bridge_freq_marks, x_marks):
        plt.axvline(x=xm, linestyle="--", linewidth=1)
        plt.text(xm, 1.05, f"{bf:g} Hz", rotation=90, va="bottom", ha="center", fontsize=9)

    plt.xlabel(r"Normalized Bridge Frequency ($f_h/f_s$)")
    plt.ylabel("Force Modification Factor" if model["invert"] else model["factor_key"])
    plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 4. One wrapper function for everything
# =========================================================
def build_force_factor_model_from_pickle(
    pkl_path,
    densities=None,
    bridge_frequencies=None,
    factor_key="factor_rms",
    invert=True,
    mass_ratio_round=3,
    use_abs=False
):
    """
    End-to-end wrapper:
    load pickle -> compute factors_all -> build interpolator

    Returns
    -------
    factors_all : list of dict
    model : dict
    """
    factors_all, results_all = compute_factors_from_pickle(
        pkl_path=pkl_path,
        densities=densities,
        bridge_frequencies=bridge_frequencies,
        use_abs=use_abs
    )

    model = build_force_factor_interpolator(
        factors_all=factors_all,
        factor_key=factor_key,
        invert=invert,
        mass_ratio_round=mass_ratio_round
    )

    return factors_all, model