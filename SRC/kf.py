"""
kf.py
=====
Python / NumPy conversion of ``src/LibKf/kf.cc``.

Contains three Kalman-filter update variants plus the public ``filter()``
dispatcher, all translated directly from the C++ original.

Functions
---------
filter_standard(x, P, H, v, R, qc, res)
    Standard KF update using the Joseph-form covariance equation.
    Corresponds to ``filter_()`` in kf.cc.

filter_vbakf(x, P, H, v, R)
    Variational-Bayes adaptive Kalman filter.
    Corresponds to ``vbakf_()`` in kf.cc.
    Reference: "A Variational Bayesian-Based Robust Adaptive Filtering for
    Precise Point Positioning Using Undifferenced and Uncombined
    Observations".

filter_sage_husa(x, P, H, v, R)
    Sage-Husa adaptive KF.
    Corresponds to ``sage_husa_()`` in kf.cc.

filter(x, P, H, v, R, qc, kf_type, res, tc)
    Public dispatcher — compresses zero states, calls the requested
    variant, and writes results back into the full arrays.
    Corresponds to ``filter()`` in kf.cc.

Notes
-----
*  All matrices are stored in **column-major** (Fortran) order in the
   original C code (``mat(n,m)`` → n rows, m cols, column-major).
   In NumPy the equivalent is a (n, m) C-order array; the arithmetic is
   identical because we use ``@`` (matmul) and ``np.linalg.inv``.

*  The original code modifies ``x`` and ``P`` in-place.  This Python
   version also modifies them in-place and returns a status code (0 = OK,
   non-zero = singular matrix error).

*  Zero states are temporarily compressed before the update to save
   computation time, exactly as in the C++ version.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from constants import KFOPT_STD, KFOPT_VBKF, KFOPT_SAGE_HUSA, TDISTB_0250, TDISTB_0005
from structures import Res

# ---------------------------------------------------------------------------
# Helper: safe square root
# ---------------------------------------------------------------------------
def _sqrt(x: float) -> float:
    return 0.0 if x <= 0.0 else float(np.sqrt(x))


# ---------------------------------------------------------------------------
# Standard Kalman filter (Joseph form)  — filter_() in kf.cc
# ---------------------------------------------------------------------------
def filter_standard(
    x: np.ndarray, P: np.ndarray,
    H: np.ndarray, v: np.ndarray, R: np.ndarray,
    qc: int = 0, res: Optional[Res] = None
) -> int:
    """Standard KF update.

    K = P·H·(H'·P·H + R)⁻¹
    x_new = x + K·v
    P_new = (I − K·H')·P·(I − K·H')' + K·R·K'   (Joseph form)

    Parameters
    ----------
    x, P : np.ndarray  — state vector (n,) and covariance (n×n), modified in-place.
    H    : np.ndarray  — design matrix transposed (n×m).
    v    : np.ndarray  — innovation vector (m,).
    R    : np.ndarray  — measurement noise covariance (m×m).
    qc   : int         — quality-control flag (currently unused inside this fn).
    res  : Res | None  — if provided, posterior residuals are written here.

    Returns
    -------
    int — 0 on success, non-zero if matrix inversion failed.
    """
    n, m = H.shape[0], H.shape[1]

    # Q = H'·P·H + R
    F = P @ H          # (n×m)
    Q = H.T @ F + R    # (m×m)

    try:
        Q_inv = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        return -1

    K  = F @ Q_inv                    # gain  (n×m)
    dx = K @ v
    # ---- SAFETY: limit update ----
    if np.linalg.norm(dx) > 10:   # 10 meters
       return -2   # reject update

    x += dx                        # state update
    I_KH = np.eye(n) - K @ H.T       # (I − K·H')
    P1 = I_KH @ P                     # (I − K·H')·P
    P2 = P1 @ I_KH.T                  # (I − K·H')·P·(I − K·H')' — copy
    P[:] = P2 + K @ R @ K.T           # Joseph form

    if res is not None:
        # post-fit residual covariance  Qvv = R · Q⁻¹ · R
        RQ   = R @ Q_inv
        res.Qvv   = RQ @ R
        # post-fit residuals  v_post = −R·Q⁻¹·v
        res.post_v = -(RQ @ v)
        # unit-weight RMS  σ₀ = √(v'·Q⁻¹·v / m)
        quad    = float(v @ (Q_inv @ v))
        res.sigma0 = _sqrt(quad / m)

    return 0


# ---------------------------------------------------------------------------
# Variational Bayes adaptive KF  — vbakf_() in kf.cc
# ---------------------------------------------------------------------------
def filter_vbakf(
    x: np.ndarray, P: np.ndarray,
    H: np.ndarray, v: np.ndarray, R: np.ndarray
) -> int:
    """Variational-Bayes adaptive Kalman filter (VBAKF).

    Implements a 10-iteration VB loop with outlier down-weighting based
    on normalised posterior residuals and a Student-t table threshold.

    Parameters
    ----------
    x, P : in-place modified state / covariance.
    H    : design matrix transposed (n×m).
    v    : innovation (m,).
    R    : measurement noise covariance (m×m).

    Returns
    -------
    int — 0 on success, non-zero on singular matrix.
    """
    n, m = H.shape[0], H.shape[1]
    N_iter = 10
    tao_P  = 3.0
    info   = 0

    # ---- Compute robust measurement covariance RI -------------------------
    # Prior post-fit residual proxy: v_post_ = R · (H'PH+R)⁻¹ · R
    F  = P @ H
    Q  = H.T @ F + R
    try:
        Q_inv = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        return -1

    v_post_ = R @ Q_inv @ R          # (m×m)
    v_post  = -(v_post_ @ v)         # (m,) — proxy posterior residuals

    RI  = R.copy()
    v_N = np.zeros(m)
    for j in range(m):
        fabs_v  = abs(v_post[j])
        denom   = _sqrt(v_post_[j, j] * RI[j, j])
        v_N[j]  = fabs_v / denom if denom > 0 else 0.0

    half = m // 2
    v_all1 = v_N[0::2].sum()   # phase channels (even indices)
    v_all2 = v_N[1::2].sum()   # pseudorange channels (odd indices)

    V_all1 = sum((v_N[2 * jj] - v_all1 / half) ** 2 for jj in range(half))
    V_all2 = sum((v_N[2 * jj + 1] - v_all2 / half) ** 2 for jj in range(half))

    max_df = min(half, len(TDISTB_0250) - 1)
    thres_lo = TDISTB_0250[max_df]
    thres_hi = TDISTB_0005[max_df]

    for j in range(m):
        if j % 2 == 0:   # phase
            mean = v_all1 / half
            var  = V_all1 / half
        else:             # pseudorange
            mean = v_all2 / half
            var  = V_all2 / half
        T_j = abs(v_N[j] - mean) / _sqrt(var) if var > 0 else 0.0
        if T_j > thres_hi:
            RI[j, j] *= 1.0e7    # reject
        elif T_j > thres_lo:
            scale    = T_j / thres_lo * ((thres_hi - thres_lo) / (thres_hi - T_j)) ** 2
            RI[j, j] *= scale    # down-weight

    # ---- VB iteration loop ------------------------------------------------
    xk1k  = x.copy()
    Pk1k  = P.copy()
    tk1k  = n + 1 + tao_P
    Tk1k  = tao_P * Pk1k

    xp  = xk1k.copy()
    Pp  = Pk1k.copy()

    for _ in range(N_iter):
        Ak  = Pp.copy()
        dx  = xp - xk1k
        Ak += np.outer(dx, dx)

        tkk = tk1k + 1
        Tkk = Tk1k + Ak

        try:
            Tkk_inv = np.linalg.inv(Tkk)
        except np.linalg.LinAlgError:
            info = -1; break

        E_i_Pk1k = (tkk - n - 1) * Tkk_inv    # expected inverse covariance

        try:
            D_Pk1k = np.linalg.inv(E_i_Pk1k)
        except np.linalg.LinAlgError:
            info = -1; break

        Pzzk1k = RI + H.T @ D_Pk1k @ H
        Pxzk1k = D_Pk1k @ H

        try:
            Pzzk1k_inv = np.linalg.inv(Pzzk1k)
        except np.linalg.LinAlgError:
            info = -1; break

        Kk   = Pxzk1k @ Pzzk1k_inv
        xp   = xk1k + Kk @ v
        Kk_  = Kk @ H.T
        Pp   = D_Pk1k - Kk_ @ D_Pk1k

    x[:] = xp
    P[:] = Pp
    return info


# ---------------------------------------------------------------------------
# Sage-Husa adaptive KF  — sage_husa_() in kf.cc
# ---------------------------------------------------------------------------
_sage_beta: float = 1.0   # module-level state (mirrors C static variable)


def filter_sage_husa(
    x: np.ndarray, P: np.ndarray,
    H: np.ndarray, v: np.ndarray, R: np.ndarray
) -> int:
    """Sage-Husa adaptive Kalman filter.

    Adaptively estimates the measurement noise covariance using an
    exponential forgetting factor ``beta``.

    NOTE: ``beta`` is maintained as module-level state (mirrors the
    ``static double beta`` in the C++ function).  Reset it to 1.0
    between independent processing runs if needed.

    Parameters
    ----------
    x, P : in-place modified state / covariance.
    H    : design matrix transposed (n×m).
    v    : innovation (m,).
    R    : measurement noise covariance (m×m).

    Returns
    -------
    int — 0 on success, non-zero on singular matrix.
    """
    global _sage_beta
    n, m = H.shape[0], H.shape[1]
    info = 0

    R_ = R.copy()

    Pxykk_1 = P @ H             # (n×m)
    Py0      = H.T @ Pxykk_1    # (m×m)  = H'·P·H
    ykk_1    = H.T @ x          # (m,)
    rk       = v - ykk_1        # actual innovation residual

    for i in range(m):
        if v[i] > 1e10:
            continue
        ry = rk[i] ** 2 - Py0[i, i]
        R_[i, i] = (1.0 - _sage_beta) * R_[i, i] + _sage_beta * ry

    _sage_beta = _sage_beta / (_sage_beta + 0.5)

    Pykk_1    = Py0 + R_        # (m×m)
    Pykk_1_cp = Pykk_1.copy()

    try:
        Pykk_1_inv = np.linalg.inv(Pykk_1)
    except np.linalg.LinAlgError:
        return -1

    Kk   = Pxykk_1 @ Pykk_1_inv   # (n×m)
    x[:] += Kk @ rk
    P[:] -= (Kk @ Pykk_1_cp) @ Kk.T

    return info


# ---------------------------------------------------------------------------
# Public dispatcher  — filter() in kf.cc
# ---------------------------------------------------------------------------
def filter(
    x: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    v: np.ndarray,
    R: np.ndarray,
    qc:      int = 0,
    kf_type: int = KFOPT_STD,
    res: Optional[Res] = None,
    tc:  int = 0
) -> int:
    """Kalman filter update dispatcher.

    Selects zero states, compresses arrays, calls the requested KF
    variant, and writes results back — exactly as the C++ ``filter()``.

    Parameters
    ----------
    x        : np.ndarray (n,)    — state vector, modified in-place.
    P        : np.ndarray (n,n)   — covariance matrix, modified in-place.
    H        : np.ndarray (n,m)   — design matrix transposed.
    v        : np.ndarray (m,)    — innovation vector.
    R        : np.ndarray (m,m)   — measurement noise covariance.
    qc       : int                — quality-control flag.
    kf_type  : int                — filter type (KFOPT_*).
    res      : Res | None         — residual container (optional).
    tc       : int                — tightly-coupled flag (debug print).

    Returns
    -------
    int — 0 on success, non-zero on failure.
    """
    n = len(x)
    m = len(v)

    # --- Build list of active (non-zero) state indices --------------------
    ix = np.array([i for i in range(n)
                   if x[i] != 0.0 and P[i, i] > 0.0], dtype=int)
    k  = len(ix)
    if k == 0:
        return -1

    # --- Compress arrays to active states ---------------------------------
    x_ = x[ix].copy()
    P_ = P[np.ix_(ix, ix)].copy()
    H_ = H[ix, :].copy()

    if tc:
        print("coupled prior:")
        print("v  =", v)
        print("H_ =\n", H_)
        print("x_ =", x_)
        print("P_ =\n", P_)

    # --- Call selected KF variant -----------------------------------------
    xp_ = x_.copy()
    Pp_ = P_.copy()

    if kf_type == KFOPT_VBKF:
        info = filter_vbakf(xp_, Pp_, H_, v, R)
    elif kf_type == KFOPT_SAGE_HUSA:
        info = filter_sage_husa(xp_, Pp_, H_, v, R)
    else:
        info = filter_standard(xp_, Pp_, H_, v, R, qc=qc, res=res)

    if tc:
        print("coupled post:")
        print("R  =\n", R)
        print("xp_=", xp_)
        print("Pp_=\n", Pp_)

    # --- Write back to full arrays ----------------------------------------
    x[ix]         = xp_
    P[np.ix_(ix, ix)] = Pp_

    return info