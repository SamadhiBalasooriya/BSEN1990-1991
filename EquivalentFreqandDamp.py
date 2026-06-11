import math
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eig, eigvals,ordqz, solve, cholesky,cholesky, solve_triangular, eigvals 
from matrix import bridge
from solver import MatrixAssemblesymetric_social
from pedestrian import Pedestrian

'''def modal_step(M, K, C, numped, num_modes):
    """
    calculates effective frequency and damping ratio using eigenvalue analysis
    One step: return (lam_pos, f_pos, zeta_pos)"""
   
    n = M.shape[0]
    Ag = np.block([[np.zeros((n,n)), np.eye(n)],
                   [-K,              -C      ]])
    Bg = np.block([[np.eye(n),       np.zeros((n,n))],
                   [np.zeros((n,n)), M      ]])
    
    #MinvK = solve(M, K, assume_a='pos', check_finite=False)   # ≈ M^{-1}K
    #MinvC = solve(M, C, assume_a='pos', check_finite=False)   # ≈ M^{-1}C
    #A = np.block([[np.zeros((n,n)), np.eye(n)],
    #              [-MinvK,          -MinvC     ]])
    # 2) QZ with left-half-plane ordering (stable eigenvalues first)
    #_, _, alpha, beta, _, _= ordqz(Ag, Bg, sort='lhp', output='complex', check_finite=False)
    #lam = alpha / beta  # already ordered: stable first

    lam, _ = eig(Ag, Bg, check_finite=False)
    #lam, _ = eig(A, check_finite=False)  # (2n,)
    keep_conj = lam.imag > 0                        # one from each conjugate pair
    lam = lam[keep_conj]
    wn = np.abs(lam)                             # rad/s
    f = wn/(2*np.pi)                             # Hz
    zeta = np.abs(lam.real)/wn                    # damping ratio

    return lam[:num_modes], f[:num_modes], zeta[:num_modes]'''

def modal_step(M, K, C, numped, num_modes,f_h, f_bridge):
    """
    Stable second-order eigen solve:
      1) Cholesky scale to make M -> I
      2) State-space with B = I
    Returns lam,f,zeta for the first `num_modes` (by ascending frequency).
    """
    # ensure arrays
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    n = M.shape[0]

    # --- 1) Cholesky of M (with tiny jitter if ever needed) ---
    try:
        L = cholesky(M, lower=True, check_finite=False)
    except Exception:
        eps = 1e-12 * (np.trace(M) / n)
        L = cholesky(M + eps * np.eye(n), lower=True, check_finite=False)

    # helper: congruence transform A -> L^{-1} A L^{-T} without forming inverses
    def cong(L, A):
        X = solve_triangular(L, A, lower=True, check_finite=False)
        return solve_triangular(L.T, X.T, lower=False, check_finite=False).T

    # scaled (mass-normalized) matrices
    Khat = cong(L, K)
    Chat = cong(L, C)

    # --- 2) State-space with Mhat = I ---
    Z = np.zeros((n, n))
    I = np.eye(n)
    Ahat = np.block([[Z,  I],
                     [-Khat, -Chat]])

    # standard eigenvalues of Ahat (B = I)
    lam = eigvals(Ahat, check_finite=False)

    # keep one from each conjugate pair (positive imag)
    keep = lam.imag > 0
    lam = lam[keep]

    wn = np.abs(lam)               # rad/s
    f  = wn / (2.0 * np.pi)        # Hz
    zeta = np.abs(lam.real) / wn   # damping ratio

    # -------- threshold-based selection (replace the old argsort block) --------
    f_r = np.asarray(f).real
    N = f_r.size

    if f_h < f_bridge:
        # pick just ABOVE f_h (ascending)
        cand = np.where(f_r > f_h)[0]
        idx = cand[np.argsort(-f_r[cand])][:num_modes]
    else:
        # pick just BELOW f_h (descending)
        cand = np.where(f_r < f_h)[0]
        idx = cand[np.argsort(f_r[cand])][:num_modes]

    #sel = ordered[:num_modes]

    return lam[idx], f[idx], zeta[idx]

'''def modal_step(M, K, C, numped, num_modes):
    """
    Return the num_modes eigenpairs that are most 'bridge-like'.
    Works well even through avoided crossings.

    N_bridge  : number of bridge DOFs (first block in your assembled ordering)
    num_modes : how many bridge-dominated modes to return (usually 1)
    """
    M = np.asarray(M, float); K = np.asarray(K, float); C = np.asarray(C, float)
    n = M.shape[0]

    # --- Cholesky mass-normalize: M = L L^T ---
    try:
        L = cholesky(M, lower=True, check_finite=False)
    except Exception:
        eps = 1e-12 * (np.trace(M)/n)
        L = cholesky(M + eps*np.eye(n), lower=True, check_finite=False)

    # congruence transform A -> L^{-1} A L^{-T}
    def cong(L, A):
        X = solve_triangular(L, A, lower=True, check_finite=False)
        return solve_triangular(L.T, X, lower=False, check_finite=False)

    Khat = cong(L, K)
    Chat = cong(L, C)

    Z = np.zeros((n, n)); I = np.eye(n)
    Ahat = np.block([[Z, I], [-Khat, -Chat]])

    # --- coupled eigenpairs ---
    lam, V = eig(Ahat, check_finite=False)   # V columns are eigenvectors
    # keep one from each complex pair
    keep = lam.imag > 0
    lam = lam[keep]; V = V[:, keep]

    # position part in mass-normalized coordinates (y = L^T x)
    Vpos = V[:n, :]               # shape (n, m)
    # bridge fraction: mass-normalized kinetic-energy fraction in bridge DOFs
    # (Mhat = I => mass-weighted norm is just Euclidean)
    num = np.sum(np.abs(Vpos[:num_modes, :])**2, axis=0)
    den = np.sum(np.abs(Vpos)**2, axis=0) + 1e-300
    bridge_frac = num / den       # in [0,1]

    # order modes by 'bridge-likeness' (highest first)
    order = np.argsort(-bridge_frac)
    sel = order[:num_modes]

    lam_sel = lam[sel]
    wn = np.abs(lam_sel)                 # rad/s
    f  = wn/(2*np.pi)                    # Hz
    zeta = np.abs(lam_sel.real)/wn       # damping ratio (stable LHP)

    return lam_sel, f, zeta'''

def compute_accelerance(M, K, C, omega):
    """
    Compute accelerance FRF H(ω) for system (M,C,K).
    - M, K, C: (n+N, n+N) matrices
    - omega: array of angular frequencies (rad/s)
    Returns: H of shape (len(omega), n+N, n+N)
    """
    n = M.shape[0]
    H = np.empty((len(omega), n, n), dtype=complex)

    I = np.eye(n)   # identity for RHS

    for k, w in enumerate(omega):
        # Dynamic stiffness
        Z = K - (w**2)*M + 1j*w*C
        # Instead of inv(Z), solve Z X = I
        H[k] = -w**2 * solve(Z, I)
    return H



'''def modal_step(M, K, C, numped, num_modes, fmin_hz=0.1, fmax_hz=np.inf):
    """
    Stable modes from QEP: (λ^2 M + λ C + K)φ = 0
    - Mass-normalize with Cholesky to avoid generalized eig
    - Keep one eigenvalue per complex-conjugate pair (imag < 0 or real < 0 if purely real)
    - Sort by frequency and return the first num_modes
    """
    # (1) Enforce symmetry softly (mitigate tiny assembly asymmetries)
    M = 0.5*(M + M.T);  K = 0.5*(K + K.T);  C = 0.5*(C + C.T)

    # (2) Mass-normalize: M -> I, improves conditioning
    n = M.shape[0]
    L  = cholesky(M, lower=True, check_finite=False)        # M = L L^T
    Li = solve(L, np.eye(n), lower=True, check_finite=False)  # L^{-1}
    Kt = Li @ K @ Li.T
    Ct = Li @ C @ Li.T

    # (3) Standard state matrix (no generalized eig)
    A = np.block([[np.zeros((n,n)), np.eye(n)],
                  [-Kt,            -Ct     ]])

    lam = eig(A, check_finite=False)[0]   # eigenvalues only

    # (4) Keep one per conjugate pair (lower half-plane) + purely real negatives
    #keep = (lam.imag < 0) | (np.isclose(lam.imag, 0.0) & (lam.real < 0))
    #lam  = lam[keep]

    # (5) Natural freq & damping ratio (stable => ζ >= 0)
    wn   = np.abs(lam)                    # rad/s
    f    = wn / (2*np.pi)                 # Hz
    zeta = -lam.real / wn                 # ζ = -Re(λ)/|λ|

    # (6) Band/quality filters to drop rigid-body/noise
    good = np.isfinite(f) & np.isfinite(zeta) & (f >= fmin_hz) & (f <= fmax_hz) & (zeta >= 0)
    f, zeta, lam = f[good], zeta[good], lam[good]

    # (7) Sort by frequency and take first num_modes
    #ord_ = np.argsort(f)
    #ord_ = ord_[:min(num_modes, ord_.size)]
    return lam[:num_modes], f[:num_modes], zeta[:num_modes]'''




def track_modes(prev_f, curr_f):
    """Permutation that matches current modes to previous by nearest frequency."""
    # handle NaNs defensively
    p = np.nan_to_num(prev_f, nan=np.inf)
    c = np.nan_to_num(curr_f, nan=np.inf)
    C = (p[:, None] - c[None, :])**2             # cost
    r, perm = linear_sum_assignment(C)
    return perm


def compute_series_for_one(Human, Bridge, mped, kped, cped,
                           xrb_2d, length, modalmass, numbers, numped,
                           t, func_list, pedBodyF, beamFreq):
    T = xrb_2d.shape[0]

    
    M0, K0, C0, _ = MatrixAssemblesymetric_social(Human, Bridge, mped, kped, cped,
                                                  xrb_2d[0, :], length, modalmass,
                                                  numbers, numped, t[0], func_list)
    lam0, f0, zeta0 = modal_step(M0, K0, C0, numped, numbers, pedBodyF, beamFreq)
    n_keep = min(numbers, len(f0)) if len(f0) > 0 else 0
    if n_keep == 0:
        raise RuntimeError("No modal poles found.")

    f_series    = np.full((T, n_keep), np.nan)
    zeta_series = np.full((T, n_keep), np.nan)
    lam_series  = np.full((T, n_keep), np.nan+1j*np.nan)

    f_series[0, :], zeta_series[0, :], lam_series[0, :] = f0[:n_keep], zeta0[:n_keep], lam0[:n_keep]
    prev_f = f_series[0, :].copy()

    for i in range(1, T):
        M, K, C, _ = MatrixAssemblesymetric_social(Human, Bridge, mped, kped, cped,
                                                   xrb_2d[i, :], length, modalmass,
                                                   numbers, numped, t[i], func_list)
        lam_k, f_k, zeta_k = modal_step(M, K, C, numped, numbers, pedBodyF, beamFreq)

        if len(f_k) < n_keep:
            f_k    = np.pad(f_k,    (0, n_keep-len(f_k)),    constant_values=np.nan)
            zeta_k = np.pad(zeta_k, (0, n_keep-len(zeta_k)), constant_values=np.nan)
            lam_k  = np.pad(lam_k,  (0, n_keep-len(lam_k)),  constant_values=np.nan+1j*np.nan)
        f_k, zeta_k, lam_k = f_k[:n_keep], zeta_k[:n_keep], lam_k[:n_keep]

        if np.all(np.isfinite(prev_f)) and np.all(np.isfinite(f_k)):
            perm = track_modes(prev_f, f_k)
            f_series[i, :], zeta_series[i, :], lam_series[i, :] = f_k[perm], zeta_k[perm], lam_k[perm]
            prev_f = f_series[i, :].copy()
        else:
            f_series[i, :], zeta_series[i, :], lam_series[i, :] = f_k, zeta_k, lam_k
            prev_f = f_k.copy()

    return f_series, zeta_series, lam_series


def compute_series_for_all_trials(Human, Bridge, mped, kped, cped,
                                  xrb_3d, length, modalmass, numbers,
                                  t, func_list, pedBodyF, beamFreq):
    R, T, P = xrb_3d.shape
    numped = P
    f_stack    = []
    zeta_stack = []
    lam_stack  = []
    for r in range(R):
        f_ser, z_ser, lam_ser = compute_series_for_one(
            Human, Bridge, mped, kped, cped,
            xrb_3d[r], length, modalmass, numbers, numped,
            t, func_list, pedBodyF, beamFreq
        )
        f_stack.append(f_ser); zeta_stack.append(z_ser); lam_stack.append(lam_ser)
    return np.stack(f_stack), np.stack(zeta_stack), np.stack(lam_stack)   # shapes (R, T, n_keep)

def qoi_from_series(f_series, zeta_series, mode=0, agg="mean"):
    if agg == "mean":
        return float(np.nanmean(f_series[:, mode])), float(np.nanmean(zeta_series[:, mode]))
    elif agg == "rms":
        f = f_series[:, mode]; z = zeta_series[:, mode]
        return float(np.sqrt(np.nanmean(f**2))), float(np.sqrt(np.nanmean(z**2)))
    elif agg == "peak":
        return float(np.nanmax(f_series[:, mode])), float(np.nanmax(zeta_series[:, mode]))
    else:
        raise ValueError("agg must be 'mean', 'rms', or 'peak'")

def run_one_sobol_sample(args):
    (
        length, f_h, xi_h, mped_scalar,
        pedpace, pedphase, pedvelocity,
        xrb_3d, Bridge, t, func_list,
        beamFreq_fixed, modalmass, numbers
    ) = args

    R, T, P = xrb_3d.shape
    mped_vec = np.full(P, mped_scalar)
    Fvalues = np.full(P, f_h)
    xivalues = np.full(P, xi_h)

    kped = (2 * np.pi * Fvalues) ** 2 * mped_vec
    cped = (2 * np.pi * Fvalues) * 2 * xivalues * mped_vec

    Human = Pedestrian(
        mass=mped_vec,
        damp=cped,
        stiff=kped,
        pace=pedpace,
        phase=pedphase,
        location=None,
        velocity=pedvelocity,
        iSync=0.3,
    )

    f_stack, zeta_stack, _ = compute_series_for_all_trials(
        Human, Bridge, mped_vec, kped, cped,
        xrb_3d, length, modalmass, numbers,
        t, func_list,
        pedBodyF=f_h, beamFreq=beamFreq_fixed
    )

    f_per = np.nanmean(f_stack[:, :, 0], axis=1)   # one value per trial
    z_per = np.nanmean(zeta_stack[:, :, 0], axis=1)
    return f_per.mean(), z_per.mean(), f_per, z_per    

def _reduce_over_time(series_stack, mode_idx=0, agg="mean"):
    """
    series_stack: (R, T, n_keep) array from compute_series_for_all_trials
    Returns: per-trial scalars (R,)
    """
    x = series_stack[:, :, mode_idx]  # (R, T)
    if agg == "mean":
        return np.nanmean(x, axis=1)
    elif agg == "rms":
        return np.sqrt(np.nanmean(x**2, axis=1))
    elif agg == "peak":
        return np.nanmax(x, axis=1)
    else:
        raise ValueError("agg must be 'mean' | 'rms' | 'peak'")