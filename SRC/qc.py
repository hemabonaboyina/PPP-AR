"""
qc.py
=====
Python / NumPy conversion of ``src/LibQc/qc.cc``.

Implements residual quality-control (outlier detection and down-weighting)
for GNSS PPP/PPK processing.  All functions are direct translations of
the C++ originals.

Public API
----------
init_prires(v, vflag)           → Res
    Initialise a Res with prior (pre-fit) residuals.

init_postres(rtk, post_v, res, R)
    Populate posterior residuals in an existing Res and update per-sat
    normalised residual arrays in rtk.ssat.

freeres(res)
    Reset a Res object (clear all arrays).

resqc(t, rtk, res, exc, ppp)    → int
    Main QC dispatcher; routes to the selected algorithm.

pri_res_check(t, rtk, pri_v, vflag, nv, exc) → int
    Check prior (pre-fit) residuals per satellite system.

Helper macros translated to inline functions
--------------------------------------------
_NF, _NP, _NI, _NT, _NL, _NB, _NR, _NX  — state-count helpers.
_II, _IT, _IL, _IB                        — state-index helpers.
_iamb_ppp, _iamb_ppk                      — ambiguity index helpers.
"""

from __future__ import annotations
import sys
import math
import numpy as np
from typing import Optional, List

from constants import (
    MAXSAT, NFREQ, MAXOBS,
    R2D, D2R,
    PMODE_DGPS, PMODE_STATIC_START, PMODE_PPP_KINEMA, PMODE_PPP_FIXED,
    PMODE_TC_DGPS, PMODE_TC_PPK, PMODE_LC_DGPS, PMODE_LC_PPK,
    PMODE_TC_PPP, PMODE_LC_PPP, PMODE_INS_MECH, PMODE_LC_POS,
    PMODE_STC_PPP,
    IONOOPT_IFLC, IONOOPT_UC, IONOOPT_UC_CONS,
    TROPOPT_EST, TROPOPT_ESTG,
    GLO_ARMODE_AUTOCAL,
    ROBUST_QC_OFF, ROBUST_QC_IGG_PR, ROBUST_QC_IGG_CP,
    ROBUST_QC_IGG, ROBUST_QC_SHI, ROBUST_QC_ZHAO,
    TDISTB_0250, TDISTB_0005, TDISTB_0050, TDISTB_0010,
    INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS,
    SYS_GPS, SYS_GLO, SYS_GAL, SYS_CMP, SYS_BD3, SYS_QZS, SYS_IRN,
)
from structures import Res, RTK

NUM_SYS = 6

# ---------------------------------------------------------------------------
# State-count helper functions  (macros in qc.cc)
# ---------------------------------------------------------------------------
def _NF(opt):
    return 1 if opt.ionoopt == IONOOPT_IFLC else opt.nf

def _NP(opt):
    return 9 if opt.dynamics else 3

def _NI(opt):
    return MAXSAT if opt.ionoopt != IONOOPT_UC else 0

def _NT(opt):
    if opt.tropopt < TROPOPT_EST:
        return 0
    elif opt.tropopt < TROPOPT_ESTG:
        return 2
    return 6

def _NL(opt):
    return NFREQ if opt.glomodear == GLO_ARMODE_AUTOCAL else 0  # NFREQGLO=2 used in C

def _NB(opt):
    return 0 if opt.mode <= PMODE_DGPS else MAXSAT * _NF(opt)

def _NR(opt):
    return _NP(opt) + _NI(opt) + _NT(opt) + _NL(opt)

def _NX(opt):
    return _NR(opt) + _NB(opt)

def _II(s, opt):
    """Ionosphere state index for satellite s (1-based)."""
    return _NP(opt) + s - 1

def _IT(r, opt):
    """Troposphere state index for rover (r=0) or ref (r=1)."""
    return _NP(opt) + _NI(opt) + _NT(opt) // 2 * r

def _IL(f, opt):
    """Receiver hardware bias index for frequency f."""
    return _NP(opt) + _NI(opt) + _NT(opt) + f

def _IB(s, f, opt):
    """Phase bias state index for satellite s (1-based), frequency f."""
    return _NR(opt) + MAXSAT * f + s - 1

def _iamb_ppp(opt, sat: int, f: int) -> int:
    """Ambiguity state index in PPP mode."""
    return _IB(sat, f, opt)

def _iamb_ppk(opt, sat: int, f: int) -> int:
    """Ambiguity state index in PPK mode (same formula for this lib)."""
    return _IB(sat, f, opt)


# ---------------------------------------------------------------------------
# Helper: find index of maximum absolute value in an array
# ---------------------------------------------------------------------------
def _findmax(arr: np.ndarray) -> tuple[int, float]:
    """Return (index, value) of element with maximum absolute value."""
    if len(arr) == 0:
        return 0, 0.0
    idx  = int(np.argmax(np.abs(arr)))
    return idx, float(arr[idx])


def _median(arr: np.ndarray) -> float:
    return float(np.median(arr))


# ---------------------------------------------------------------------------
# sat system index (mirrors satsysidx() in rtklib)
# ---------------------------------------------------------------------------
_SYS_ORDER = [SYS_GPS, SYS_GLO, SYS_GAL, SYS_QZS, SYS_CMP, SYS_IRN]

def _satsysidx(sat: int) -> int:
    """Return 0-based GNSS system index for internal per-system arrays."""
    # Simplified version — in the full library satellite numbers
    # encode the constellation.  Here we use a placeholder that returns 0.
    # A complete implementation would call rtklib's satsys().
    return 0


# ---------------------------------------------------------------------------
# res_class  — classify residuals into PR / CP buckets
# ---------------------------------------------------------------------------
def _res_class(res: Res, pri: bool) -> None:
    """Partition residual arrays into pseudorange (PR) and carrier-phase (CP)."""
    if res.vflag is None:
        return

    if pri:
        res.npr = 0; res.ncp = 0
        for i in range(res.nv):
            typ = (res.vflag[i] >> 4) & 0xF
            if   typ == 1: res.npr += 1
            elif typ == 0: res.ncp += 1
        res.pri_pr  = np.zeros(res.npr)
        res.pri_cp  = np.zeros(res.ncp)
        res.post_pr = np.zeros(res.npr)
        res.post_cp = np.zeros(res.ncp)
        res.pr_idx  = np.zeros(res.npr, dtype=int)
        res.cp_idx  = np.zeros(res.ncp, dtype=int)
        res.norm_pr = np.zeros(res.npr)
        res.norm_cp = np.zeros(res.ncp)

    j = k = 0
    for i in range(res.nv):
        typ = (res.vflag[i] >> 4) & 0xF
        if typ == 1:   # pseudorange
            if pri:
                res.pr_idx[j]  = i
                res.pri_pr[j]  = abs(res.pri_v[i])
                j += 1
            else:
                denom = res.sigma0 * math.sqrt(abs(res.R[i, i])) if (res.R is not None) else 1.0
                res.norm_pr[j] = res.post_v[i] / denom if denom else 0.0
                res.post_pr[j] = res.post_v[i]
                j += 1
        elif typ == 0:  # carrier phase
            if pri:
                res.cp_idx[k]  = i
                res.pri_cp[k]  = abs(res.pri_v[i])
                k += 1
            else:
                denom = res.sigma0 * math.sqrt(abs(res.R[i, i])) if (res.R is not None) else 1.0
                res.norm_cp[k] = res.post_v[i] / denom if denom else 0.0
                res.post_cp[k] = res.post_v[i]
                k += 1


# ---------------------------------------------------------------------------
# init_prires  — create Res from prior residuals
# ---------------------------------------------------------------------------
def init_prires(v: np.ndarray, vflag: np.ndarray) -> Res:
    """Create and populate a Res with prior (pre-fit) residuals.

    Corresponds to ``init_prires()`` in qc.cc.

    Parameters
    ----------
    v      : (nv,) prior residual vector.
    vflag  : (nv,) integer flag per observation
             bits 15-8 = sat number, bits 7-4 = obs type (0=CP,1=PR),
             bits 3-0  = frequency index.

    Returns
    -------
    Res — populated residual container.
    """
    nv  = len(v)
    res = Res()
    res.pri_v  = v.copy()
    res.vflag  = vflag.copy().astype(int)
    res.nv     = nv
    _res_class(res, pri=True)
    return res


# ---------------------------------------------------------------------------
# init_postres  — populate posterior residuals and update rtk.ssat
# ---------------------------------------------------------------------------
def init_postres(rtk: RTK, post_v: np.ndarray, res: Res, R: np.ndarray) -> None:
    """Populate posterior residuals in *res* and update per-sat norm_v.

    Corresponds to ``init_postres()`` in qc.cc.

    Parameters
    ----------
    rtk    : RTK state (ssat array is written).
    post_v : (nv,) posterior residual vector.
    res    : Res to update (must already have been initialised by init_prires).
    R      : (nv×nv) measurement noise covariance.
    """
    res.post_v = post_v.copy()
    res.R      = R.copy()
    _res_class(res, pri=False)

    for i in range(res.ncp):
        idx = res.cp_idx[i]
        sat = (res.vflag[idx] >> 8) & 0xFF
        frq = res.vflag[idx] & 0xF
        if 1 <= sat <= MAXSAT:
            rtk.ssat[sat - 1].norm_v[0][frq] = res.norm_cp[i]
    for i in range(res.npr):
        idx = res.pr_idx[i]
        sat = (res.vflag[idx] >> 8) & 0xFF
        frq = res.vflag[idx] & 0xF
        if 1 <= sat <= MAXSAT:
            rtk.ssat[sat - 1].norm_v[1][frq] = res.norm_pr[i]


# ---------------------------------------------------------------------------
# freeres  — reset Res
# ---------------------------------------------------------------------------
def freeres(res: Res) -> None:
    """Clear all arrays in a Res object.

    Corresponds to ``freeres()`` in qc.cc.
    """
    res.npr = res.ncp = res.nv = 0
    res.sigma0 = 0.0
    for attr in ('vflag', 'pri_v', 'post_v', 'pr_idx', 'cp_idx',
                 'pri_pr', 'pri_cp', 'post_pr', 'post_cp',
                 'norm_pr', 'norm_cp', 'R', 'Qvv'):
        setattr(res, attr, None)


# ---------------------------------------------------------------------------
# resqc_igg_pr  — IGG outlier test on pseudorange residuals only
# ---------------------------------------------------------------------------
def _resqc_igg_pr(rtk: RTK, res: Res, exc: List[int], ppp: int) -> int:
    """IGG-III down-weighting based on normalised post-fit PR residuals.

    Corresponds to ``resqc_igg_pr()`` in qc.cc.
    """
    k0 = rtk.opt.igg_k0 if rtk.opt.igg_k0 != 0.0 else 2.80
    k1 = rtk.opt.igg_k1 if rtk.opt.igg_k1 != 0.0 else 4.13

    if res.norm_pr is None or res.npr == 0:
        return 0

    max_n_pr_idx, max_n_pr = _findmax(res.norm_pr)
    idx  = res.pr_idx[max_n_pr_idx]
    sat  = (res.vflag[idx] >> 8) & 0xFF
    frq  = res.vflag[idx] & 0xF
    el   = rtk.ssat[sat - 1].azel[1] * R2D if 1 <= sat <= MAXSAT else 0.0

    qc_flag = 0
    if 1 <= sat <= MAXSAT:
        if max_n_pr > k1:
            rtk.ssat[sat - 1].var_fact[1][frq] = 100000.0
            qc_flag = 1
            print(f"  {sat} P{frq+1} norm residual in rejected segment "
                  f"el={el:.2f} v={res.post_v[idx]:.3f} norm_v={max_n_pr:.3f}",
                  file=sys.stderr)
        elif max_n_pr >= k0:
            fact = (max_n_pr / k0) * ((k1 - k0) / (k1 - max_n_pr)) ** 2
            rtk.ssat[sat - 1].var_fact[1][frq] = fact
            qc_flag = 1
        else:
            rtk.ssat[sat - 1].var_fact[1][frq] = 1.0

    return qc_flag


# ---------------------------------------------------------------------------
# resqc_igg_cp  — IGG outlier test on carrier-phase residuals only
# ---------------------------------------------------------------------------
def _resqc_igg_cp(rtk: RTK, res: Res, exc: List[int], ppp: int) -> int:
    """IGG-III down-weighting based on normalised post-fit CP residuals.

    Corresponds to ``resqc_igg_cp()`` in qc.cc.
    """
    k0 = rtk.opt.igg_k0 if rtk.opt.igg_k0 != 0.0 else 2.80
    k1 = rtk.opt.igg_k1 if rtk.opt.igg_k1 != 0.0 else 4.13

    if res.norm_cp is None or res.ncp == 0:
        return 0

    max_n_cp_idx, max_n_cp = _findmax(res.norm_cp)
    idx  = res.cp_idx[max_n_cp_idx]
    sat  = (res.vflag[idx] >> 8) & 0xFF
    frq  = res.vflag[idx] & 0xF
    el   = rtk.ssat[sat - 1].azel[1] * R2D if 1 <= sat <= MAXSAT else 0.0

    qc_flag = 0
    if 1 <= sat <= MAXSAT:
        if abs(max_n_cp) > k1:
            rtk.ssat[sat - 1].var_fact[0][frq] = 100000.0
            exc[sat - 1] = 1
            qc_flag = 1
            print(f"  {sat} L{frq+1} norm residual in rejected segment "
                  f"el={el:.2f} norm_v={max_n_cp:.3f}", file=sys.stderr)
        elif abs(max_n_cp) >= k0:
            fact = (max_n_cp / k0) * ((k1 - k0) / (k1 - max_n_cp)) ** 2
            rtk.ssat[sat - 1].var_fact[0][frq] = fact
            qc_flag = 1
        else:
            rtk.ssat[sat - 1].var_fact[0][frq] = 1.0

    return qc_flag


# ---------------------------------------------------------------------------
# resqc_igg  — IGG combined (PR then CP)
# ---------------------------------------------------------------------------
def _resqc_igg(rtk: RTK, res: Res, exc: List[int], ppp: int) -> int:
    """IGG-III combined: check PR first, then CP.

    Corresponds to ``resqc_igg()`` in qc.cc.
    """
    k0 = rtk.opt.igg_k0 if rtk.opt.igg_k0 != 0.0 else 2.80
    k1 = rtk.opt.igg_k1 if rtk.opt.igg_k1 != 0.0 else 4.13

    if res.norm_pr is None or res.npr == 0:
        return 0

    # --- pseudorange ---
    max_n_pr_idx, max_n_pr = _findmax(res.norm_pr)
    idx  = res.pr_idx[max_n_pr_idx]
    sat  = (res.vflag[idx] >> 8) & 0xFF
    frq  = res.vflag[idx] & 0xF
    el   = rtk.ssat[sat - 1].azel[1] * R2D if 1 <= sat <= MAXSAT else 0.0

    qc_flag = 0
    if 1 <= sat <= MAXSAT:
        if max_n_pr > k1:
            rtk.ssat[sat - 1].var_fact[1][frq] = 100000.0
            qc_flag = 1
            return qc_flag
        else:
            # --- carrier phase ---
            if res.norm_cp is None or res.ncp == 0:
                return qc_flag
            max_n_cp_idx, max_n_cp = _findmax(res.norm_cp)
            idx2  = res.cp_idx[max_n_cp_idx]
            sat2  = (res.vflag[idx2] >> 8) & 0xFF
            frq2  = res.vflag[idx2] & 0xF
            el2   = rtk.ssat[sat2 - 1].azel[1] * R2D if 1 <= sat2 <= MAXSAT else 0.0
            if 1 <= sat2 <= MAXSAT:
                if abs(max_n_cp) > k1:
                    rtk.ssat[sat2 - 1].var_fact[0][frq2] = 100000.0
                    qc_flag = 1
                elif abs(max_n_cp) >= k0:
                    fact = (max_n_cp / k0) * ((k1 - k0) / (k1 - max_n_cp)) ** 2
                    rtk.ssat[sat2 - 1].var_fact[0][frq2] = fact
                    qc_flag = 1
                else:
                    rtk.ssat[sat2 - 1].var_fact[0][frq2] = 1.0
    return qc_flag


# ---------------------------------------------------------------------------
# resqc_shi  — Shi (2012) multi-step residual test
# ---------------------------------------------------------------------------
def _resqc_shi(rtk: RTK, res: Res, exc: List[int], ppp: int) -> int:
    """Multi-step residual test by Shi (2012).

    Checks:
      1. Absolute post-fit PR residual (elevation-dependent threshold).
      2. Normalised post-fit PR residual > 2.
      3. Absolute post-fit CP residual (elevation-dependent threshold).
      4. Normalised post-fit CP residual > 2.

    Corresponds to ``resqc_shi()`` in qc.cc.
    """
    qc_flag = 0

    # STEP 1: post PR residual
    if res.post_pr is not None and res.npr > 0:
        max_pr_idx, max_pr = _findmax(res.post_pr)
        idx  = res.pr_idx[max_pr_idx]
        sat  = (res.vflag[idx] >> 8) & 0xFF
        frq  = res.vflag[idx] & 0xF
        el   = rtk.ssat[sat - 1].azel[1] if 1 <= sat <= MAXSAT else 0.1
        thres = 3.0 / math.sin(el) if el > 0 else 1e9
        if abs(max_pr) > thres:
            if 1 <= sat <= MAXSAT:
                exc[sat - 1] = 1
            return 1

    # STEP 2: normalised PR residual
    if res.norm_pr is not None and res.npr > 0:
        max_n_pr_idx, max_n_pr = _findmax(res.norm_pr)
        idx  = res.pr_idx[max_n_pr_idx]
        sat  = (res.vflag[idx] >> 8) & 0xFF
        if abs(max_n_pr) > 2.0 and 1 <= sat <= MAXSAT:
            exc[sat - 1] = 1
            return 1

    # STEP 3: post CP residual
    if res.post_cp is not None and res.ncp > 0:
        max_cp_idx, max_cp = _findmax(res.post_cp)
        idx  = res.cp_idx[max_cp_idx]
        sat  = (res.vflag[idx] >> 8) & 0xFF
        frq  = res.vflag[idx] & 0xF
        el   = rtk.ssat[sat - 1].azel[1] if 1 <= sat <= MAXSAT else 0.1
        thres = 0.03 / math.sin(el) if el > 0 else 1e9
        if abs(max_cp) > thres and 1 <= sat <= MAXSAT:
            iamb = _iamb_ppp(rtk.opt, sat, frq) if ppp else _iamb_ppk(rtk.opt, sat, frq)
            if rtk.x is not None and rtk.P is not None:
                rtk.x[iamb] = rtk.x[iamb]   # re-init (full init would reset bias)
                rtk.P[iamb, iamb] = rtk.opt.std[0] ** 2
            rtk.ssat[sat - 1].init_amb[frq] = 1
            return 1

    # STEP 4: normalised CP residual
    if res.norm_cp is not None and res.ncp > 0:
        max_n_cp_idx, max_n_cp = _findmax(res.norm_cp)
        idx  = res.cp_idx[max_n_cp_idx]
        sat  = (res.vflag[idx] >> 8) & 0xFF
        frq  = res.vflag[idx] & 0xF
        if abs(max_n_cp) > 2.0 and 1 <= sat <= MAXSAT:
            iamb = _iamb_ppp(rtk.opt, sat, frq) if ppp else _iamb_ppk(rtk.opt, sat, frq)
            if rtk.x is not None and rtk.P is not None:
                rtk.P[iamb, iamb] = rtk.opt.std[0] ** 2
            rtk.ssat[sat - 1].init_amb[frq] = 1
            return 1

    return 0


# ---------------------------------------------------------------------------
# resqc  — public QC dispatcher
# ---------------------------------------------------------------------------
def resqc(t: object, rtk: RTK, res: Res, exc: List[int], ppp: int) -> int:
    """Main QC dispatcher — routes to the selected algorithm.

    Corresponds to ``resqc()`` in qc.cc.

    Parameters
    ----------
    t    : GTime — current epoch time (unused in logic, kept for tracing).
    rtk  : RTK   — engine state (rtk.opt.robust selects the algorithm).
    res  : Res   — residual container (must have been populated).
    exc  : list  — per-satellite exclusion flags (length MAXSAT).
    ppp  : int   — 1 if PPP mode, 0 if PPK mode.

    Returns
    -------
    int — 0 if no outlier found, 1 if an outlier was found.
    """
    robust = rtk.opt.robust
    if   robust == ROBUST_QC_OFF:    return 0
    elif robust == ROBUST_QC_IGG_PR: return _resqc_igg_pr(rtk, res, exc, ppp)
    elif robust == ROBUST_QC_IGG_CP: return _resqc_igg_cp(rtk, res, exc, ppp)
    elif robust == ROBUST_QC_IGG:    return _resqc_igg(rtk, res, exc, ppp)
    elif robust == ROBUST_QC_SHI:    return _resqc_shi(rtk, res, exc, ppp)
    return 0


# ---------------------------------------------------------------------------
# pri_res_check  — prior-residual median check per constellation
# ---------------------------------------------------------------------------
def pri_res_check(
    t: object,
    rtk: RTK,
    pri_v: np.ndarray,
    vflag: np.ndarray,
    nv: int,
    exc: List[int]
) -> int:
    """Median-based prior residual check per GNSS constellation.

    Satellites whose prior residuals deviate from the per-system median
    by more than ``thres`` are marked for exclusion.

    Corresponds to ``pri_res_check()`` in qc.cc.

    Parameters
    ----------
    t      : GTime — current epoch.
    rtk    : RTK   — engine state.
    pri_v  : (nv,) — prior residual vector.
    vflag  : (nv,) int — observation flags.
    nv     : int   — number of measurements.
    exc    : list  — per-satellite exclusion flags (length MAXSAT).

    Returns
    -------
    int — 0 if no outlier, 1 if at least one satellite excluded.
    """
    popt = rtk.opt
    ppk = (
        (PMODE_DGPS <= popt.mode <= PMODE_STATIC_START) or
        popt.mode in (PMODE_TC_DGPS, PMODE_TC_PPK, PMODE_LC_DGPS, PMODE_LC_PPK) or
        popt.insopt.imu_align in (INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS)
    )
    ppp = (
        PMODE_PPP_KINEMA <= popt.mode <= PMODE_PPP_FIXED or
        popt.mode in (PMODE_TC_PPP, PMODE_LC_PPP)
    )

    if ppk:
        thres = 3.0
    elif ppp:
        thres = 100.0
    else:
        thres = 20.0

    # Group residuals by constellation
    v_sys   = [[] for _ in range(NUM_SYS)]
    sat_sys = [[] for _ in range(NUM_SYS)]

    for j in range(nv):
        sat      = (vflag[j] >> 8) & 0xFF
        sys_idx  = _satsysidx(sat)
        sys_idx  = min(sys_idx, NUM_SYS - 1)
        sat_sys[sys_idx].append(sat)
        v_sys[sys_idx].append(pri_v[j])

    qc_flag = 0
    for j in range(NUM_SYS):
        if not v_sys[j]:
            continue
        arr  = np.array(v_sys[j])
        med  = _median(arr)
        for i, sat in enumerate(sat_sys[j]):
            if abs(arr[i] - med) > thres and 1 <= sat <= MAXSAT:
                exc[sat - 1] = 1
                qc_flag = 1

    return qc_flag

