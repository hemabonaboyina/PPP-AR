"""
structures.py
=============
Python data-class equivalents of the key C structs defined in rtklib.h
and used throughout PPP_AR.

Only the fields that are actually referenced by the converted modules
(kf, qc, match_file, ppp_ar_app) are included; the full rtklib struct
set contains hundreds of fields — those are annotated with comments
pointing to the original C definition.

All matrix / vector storage is done with numpy arrays.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from constants import (
    MAXSAT, NFREQ, MAXOBS, MAXANT, MAXSTRPATH,
    PMODE_PPP_KINEMA, ARMODE_OFF,
    IONOOPT_IFLC, TROPOPT_SAAS, EPHOPT_BRDC,
    CBIAS_OPT_BRD_TGD, BD3OPT_OFF,
    SOLQ_NONE, SOLF_LLH, KFOPT_STD, ROBUST_QC_OFF,
    SYS_GPS, TSYS_GPS,
)

# ---------------------------------------------------------------------------
# gtime_t  (rtklib.h)
# ---------------------------------------------------------------------------
@dataclass
class GTime:
    """GPS/GNSS epoch time.

    Attributes
    ----------
    time : int
        Seconds since 1970-01-01 00:00:00 UTC (Unix epoch), integer part.
    sec  : float
        Sub-second fraction [0, 1).
    """
    time: int   = 0
    sec:  float = 0.0

    def __bool__(self) -> bool:
        return self.time != 0 or self.sec != 0.0


# ---------------------------------------------------------------------------
# sol_t  (rtklib.h) — simplified
# ---------------------------------------------------------------------------
@dataclass
class Sol:
    """Single-epoch GNSS solution."""
    time:    GTime            = field(default_factory=GTime)
    rr:      np.ndarray       = field(default_factory=lambda: np.zeros(6))
    qr:      np.ndarray       = field(default_factory=lambda: np.zeros(6))
    dtr:     np.ndarray       = field(default_factory=lambda: np.zeros(6))
    stat:    int              = SOLQ_NONE
    ns:      int              = 0
    age:     float            = 0.0
    ratio:   float            = 0.0


# ---------------------------------------------------------------------------
# ssat_t  (rtklib.h) — per-satellite status
# ---------------------------------------------------------------------------
@dataclass
class SSat:
    """Per-satellite status and biases."""
    vs:       int   = 0                         # satellite valid flag
    azel:     np.ndarray = field(             # azimuth & elevation [rad]
        default_factory=lambda: np.zeros(2))
    resp:     np.ndarray = field(             # residuals of pseudorange [m]
        default_factory=lambda: np.zeros(NFREQ))
    resc:     np.ndarray = field(             # residuals of carrier phase [m]
        default_factory=lambda: np.zeros(NFREQ))
    vsat:     np.ndarray = field(             # valid satellite flag per freq
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    snr:      np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ))
    fix:      np.ndarray = field(             # ambiguity fix flag
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    slip:     np.ndarray = field(             # cycle slip flag
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    half:     np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    lock:     np.ndarray = field(             # lock counter
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    outc:     np.ndarray = field(             # out-of-lock counter
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    slipc:    np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    rejc:     np.ndarray = field(             # reject counter
        default_factory=lambda: np.zeros(NFREQ, dtype=int))
    gf:       np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ))
    mw:       np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ))
    var_fact: np.ndarray = field(             # variance scale factor [type][freq]
        default_factory=lambda: np.ones((2, NFREQ)))
    norm_v:   np.ndarray = field(             # normalised post-fit residuals [type][freq]
        default_factory=lambda: np.zeros((2, NFREQ)))
    init_amb: np.ndarray = field(             # ambiguity initialised flag
        default_factory=lambda: np.zeros(NFREQ, dtype=int))


# ---------------------------------------------------------------------------
# Residual structure  (res_t — defined in rtklib.h, used by kf + qc)
# ---------------------------------------------------------------------------
@dataclass
class Res:
    """Per-epoch residual container (prior + posterior)."""
    nv:      int              = 0        # total number of measurements
    npr:     int              = 0        # number of pseudorange measurements
    ncp:     int              = 0        # number of carrier-phase measurements
    sigma0:  float            = 0.0      # unit-weight RMS
    pri_v:   Optional[np.ndarray] = None # prior (pre-fit) residuals (nv,)
    post_v:  Optional[np.ndarray] = None # posterior (post-fit) residuals (nv,)
    vflag:   Optional[np.ndarray] = None # measurement type/sat/freq flags (nv,) int
    R:       Optional[np.ndarray] = None # measurement noise covariance (nv×nv)
    Qvv:     Optional[np.ndarray] = None # residual covariance (nv×nv)

    pr_idx:  Optional[np.ndarray] = None # indices into residual arrays for PR
    cp_idx:  Optional[np.ndarray] = None # indices into residual arrays for CP
    pri_pr:  Optional[np.ndarray] = None # prior pseudorange residuals (npr,)
    pri_cp:  Optional[np.ndarray] = None # prior carrier-phase residuals (ncp,)
    post_pr: Optional[np.ndarray] = None # posterior pseudorange residuals
    post_cp: Optional[np.ndarray] = None # posterior carrier-phase residuals
    norm_pr: Optional[np.ndarray] = None # normalised posterior PR residuals
    norm_cp: Optional[np.ndarray] = None # normalised posterior CP residuals


# ---------------------------------------------------------------------------
# insopt_t  (rtklib.h) — INS options, minimal subset
# ---------------------------------------------------------------------------
@dataclass
class InsOpt:
    """INS/GNSS integration options (minimal subset)."""
    imu_align: int = 0
    rts:       int = 0
    is_imu_samenoise: int = 1


# ---------------------------------------------------------------------------
# prcopt_t  (rtklib.h) — processing options
# ---------------------------------------------------------------------------
@dataclass
class PrcOpt:
    """GNSS/INS processing options.

    Mirrors the ``prcopt_t`` C struct from rtklib.h.  Only the fields
    referenced by the converted Python code are included; extend as needed.
    """
    mode:      int   = PMODE_PPP_KINEMA  # positioning mode
    soltype:   int   = 0                 # solution type (0=forward)
    nf:        int   = 2                 # number of frequencies
    navsys:    int   = SYS_GPS           # navigation systems bit-mask
    elmin:     float = 10.0 * (3.14159 / 180.0)  # elevation mask (rad)
    elmaskar:  float = 0.0               # elevation mask for AR (rad)
    elmaskhold: float = 0.0

    dynamics:  int   = 0                 # dynamics model (0=kinematic)
    tidecorr:  int   = 0                 # tidal correction
    ionoopt:   int   = IONOOPT_IFLC      # ionosphere option
    tropopt:   int   = TROPOPT_SAAS      # troposphere option
    sateph:    int   = EPHOPT_BRDC       # satellite ephemeris type
    cbiaopt:   int   = CBIAS_OPT_BRD_TGD # code bias option
    modear:    int   = ARMODE_OFF        # AR mode
    glomodear: int   = 0
    gpsmodear: int   = 0
    galmodear: int   = 0
    bdsmodear: int   = 0
    arprod:    int   = 0                 # AR product type
    sdopt:     int   = 0
    bd3opt:    int   = BD3OPT_OFF

    robust:    int   = ROBUST_QC_OFF     # robust QC type
    kf_type:   int   = KFOPT_STD         # KF type

    igg_k0:    float = 0.0               # IGG k0 parameter (0 → default 2.80)
    igg_k1:    float = 0.0               # IGG k1 parameter (0 → default 4.13)

    std:       np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.001, 0.0]))
    prn:       np.ndarray = field(
        default_factory=lambda: np.array([1e-4, 1e-3, 1e-4, 1e-1, 1e-2]))
    sclkstab:  float = 5e-12

    ts:        GTime = field(default_factory=GTime)   # processing start time
    te:        GTime = field(default_factory=GTime)   # processing end time

    prcdir:    str = ""    # processing root directory
    obsdir:    str = ""    # obs sub-directory name
    site_name: str = ""    # 4-char station name
    site_list: str = ""    # comma-separated site filter
    site_idx:  int = 0     # site counter

    ac_name:   str = "com"   # analysis centre name (e.g. "com", "gbm", "wum")
    aclong:    int = 1        # use long (IGS 3rd-generation) file names
    prdtype:   str = "FIN"   # product type string ("FIN", "RAP", …)
    clk_int:   str = "30S"   # clock file interval string
    eph_int:   str = "05M"   # orbit file interval string
    atx_week:  int = 2171    # IGS ATX product week

    prctype:   int = 0        # 0=single day, 1=multi-day

    insopt:    InsOpt = field(default_factory=InsOpt)


# ---------------------------------------------------------------------------
# solopt_t  (rtklib.h) — solution output options
# ---------------------------------------------------------------------------
@dataclass
class SolOpt:
    """Solution output options."""
    posf:    int   = SOLF_LLH   # position format
    times:   int   = TSYS_GPS   # time format
    timef:   int   = 0
    timeu:   int   = 3
    degf:    int   = 0
    outhead: int   = 0
    outopt:  int   = 0
    outvel:  int   = 0
    datum:   int   = 0
    height:  int   = 0
    geoid:   int   = 0
    solstatic: int = 0
    sstat:   int   = 0
    trace:   int   = 0
    ambres:  int   = 0
    nmeaintv: float = 0.0
    sep:     str   = " "
    prog:    str   = ""


# ---------------------------------------------------------------------------
# filopt_t  (rtklib.h) — file path options
# ---------------------------------------------------------------------------
@dataclass
class FilOpt:
    """File path options — paths to all ancillary product files."""
    satantp: str = ""    # satellite antenna parameters
    rcvantp: str = ""    # receiver antenna parameters
    stapos:  str = ""    # station position
    geoid:   str = ""    # geoid data
    iono:    str = ""    # ionosphere data
    dcb:     str = ""    # code bias (DCB)
    mgexdcb: str = ""    # MGEX DCB / BSX
    eop:     str = ""    # Earth rotation parameters
    blq:     str = ""    # BLQ ocean loading
    atx:     str = ""    # ATX antenna phase centre
    bia:     str = ""    # OSB bias file
    fcb:     str = ""    # FCB file

    robsf:   str = ""    # rover observation file
    bobsf:   str = ""    # base observation file
    solf:    str = ""    # solution output file
    rtsfile: str = ""    # RTS smoothing output
    rts_ins_fw: str = "" # RTS forward binary

    wl_amb:  str = ""    # wide-lane ambiguity output
    nl_amb:  str = ""    # narrow-lane ambiguity output
    lc_amb:  str = ""    # LC ambiguity output

    # Lists of up to 3 day-triplet filenames
    navf:    List[str] = field(default_factory=lambda: ["", "", ""])
    sp3f:    List[str] = field(default_factory=lambda: ["", "", ""])
    clkf:    List[str] = field(default_factory=lambda: ["", "", ""])
    updf:    List[str] = field(default_factory=lambda: ["", "", ""])


# ---------------------------------------------------------------------------
# rtk_t  (rtklib.h) — RTK/PPP engine state  (minimal, for QC module)
# ---------------------------------------------------------------------------
@dataclass
class RTK:
    """RTK/PPP engine state carrier.

    Contains the Kalman-filter state vector (``x``), covariance (``P``),
    per-satellite status array (``ssat``), processing options (``opt``),
    and current solution (``sol``).

    ``nx`` and ``na`` are the sizes of the full and fixed-solution state
    vectors respectively.
    """
    nx:    int                   = 0
    na:    int                   = 0
    tt:    float                 = 0.0
    epoch: int                   = 0
    x:     Optional[np.ndarray]  = None   # state vector (nx,)
    P:     Optional[np.ndarray]  = None   # state covariance (nx×nx)
    xa:    Optional[np.ndarray]  = None   # fixed state (na,)
    Pa:    Optional[np.ndarray]  = None   # fixed covariance (na×na)

    sol:   Sol    = field(default_factory=Sol)
    opt:   PrcOpt = field(default_factory=PrcOpt)

    ssat:  List[SSat] = field(
        default_factory=lambda: [SSat() for _ in range(MAXSAT)])

    tc:    int = 0   # tightly-coupled flag
    ins_kf: object = None  # placeholder for InsKF (not converted)