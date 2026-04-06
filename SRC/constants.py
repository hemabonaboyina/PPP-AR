"""
constants.py
============
Python equivalents of the #define constants, enums and table data found in
rtklib.h and scattered across the PPP_AR C++ source files.

All physical constants, satellite-system flags, processing-mode codes,
ambiguity-resolution product codes, observation codes, error-factor values,
solution-quality flags, GNSS frequencies, and statistical tables (Student-t
quantiles used for robust estimation) are collected here.
"""

import math

# ---------------------------------------------------------------------------
# Mathematical / physical constants
# ---------------------------------------------------------------------------
PI       = math.pi
D2R      = PI / 180.0          # degrees  → radians
R2D      = 180.0 / PI          # radians  → degrees
CLIGHT   = 299_792_458.0       # speed of light (m/s)
SC2RAD   = 3.1415926535898     # semi-circle → radian (IS-GPS)
AU       = 149_597_870_691.0   # 1 AU (m)
AS2R     = D2R / 3600.0        # arc-second → radian

OMGE     = 7.2921151467e-5     # Earth angular velocity (rad/s) IS-GPS
RE_WGS84 = 6_378_137.0        # WGS-84 semi-major axis (m)
FE_WGS84 = 1.0 / 298.257223563  # WGS-84 flattening

HION     = 350_000.0           # Ionosphere height (m)

# ---------------------------------------------------------------------------
# GNSS frequencies (Hz)
# ---------------------------------------------------------------------------
FREQ1      = 1.57542e9   # L1/E1/B1C
FREQ2      = 1.22760e9   # L2
FREQ5      = 1.17645e9   # L5/E5a/B2a
FREQ6      = 1.27875e9   # E6/L6
FREQ7      = 1.20714e9   # E5b
FREQ8      = 1.191795e9  # E5a+b
FREQ9      = 2.492028e9  # S band (IRNSS)

FREQ1_GLO  = 1.60200e9   # GLONASS G1 base
DFRQ1_GLO  = 0.56250e6   # GLONASS G1 bias per slot (Hz/n)
FREQ2_GLO  = 1.24600e9   # GLONASS G2 base
DFRQ2_GLO  = 0.43750e6   # GLONASS G2 bias per slot
FREQ3_GLO  = 1.202025e9  # GLONASS G3
FREQ1a_GLO = 1.600995e9  # GLONASS G1a
FREQ2a_GLO = 1.248060e9  # GLONASS G2a

FREQ1_CMP  = 1.561098e9  # BDS B1I
FREQ2_CMP  = 1.20714e9   # BDS B2I/B2b
FREQ3_CMP  = 1.26852e9   # BDS B3

NFREQ     = 3   # default number of carrier frequencies
NFREQGLO  = 2   # GLONASS carrier frequencies
NEXOBS    = 3   # number of extended obs codes

# ---------------------------------------------------------------------------
# Navigation system bit-masks
# ---------------------------------------------------------------------------
SYS_NONE = 0x00
SYS_GPS  = 0x01
SYS_SBS  = 0x02
SYS_GLO  = 0x04
SYS_GAL  = 0x08
SYS_QZS  = 0x10
SYS_CMP  = 0x20
SYS_BD3  = 0x40
SYS_IRN  = 0x80
SYS_LEO  = 0x100
SYS_ALL  = 0xFF

# Time systems
TSYS_GPS = 0
TSYS_UTC = 1
TSYS_GLO = 2
TSYS_GAL = 3
TSYS_QZS = 4
TSYS_CMP = 5
TSYS_IRN = 6

# ---------------------------------------------------------------------------
# Satellite PRN ranges and counts (with all optional constellations enabled)
# ---------------------------------------------------------------------------
MINPRNGPS = 1;   MAXPRNGPS = 32;  NSATGPS = MAXPRNGPS - MINPRNGPS + 1
MINPRNGLO = 1;   MAXPRNGLO = 27;  NSATGLO = MAXPRNGLO - MINPRNGLO + 1
MINPRNGAL = 1;   MAXPRNGAL = 36;  NSATGAL = MAXPRNGAL - MINPRNGAL + 1
MINPRNQZS = 193; MAXPRNQZS = 202; NSATQZS = MAXPRNQZS - MINPRNQZS + 1
MINPRNCMP = 1;   MAXPRNCMP = 50;  NSATCMP = MAXPRNCMP - MINPRNCMP + 1
MINPRNIRN = 1;   MAXPRNIRN = 7;   NSATIRN = MAXPRNIRN - MINPRNIRN + 1
MINPRNSBS = 120; MAXPRNSBS = 142; NSATSBS = MAXPRNSBS - MINPRNSBS + 1
MINPRNLEO = 1;   MAXPRNLEO = 10;  NSATLEO = MAXPRNLEO - MINPRNLEO + 1

NSYS = 7   # GPS + GLO + GAL + QZS + CMP + IRN + LEO (all enabled)

MAXSAT = (NSATGPS + NSATGLO + NSATGAL + NSATQZS
          + NSATCMP + NSATIRN + NSATSBS + NSATLEO)

MAXOBS     = 64
MAXRCV     = 64
MAXOBSTYPE = 64
MAXFILE    = 12
MAXSTRPATH = 300
MAXANT     = 64
MAXCODE    = 68

DTTOL      = 0.025   # tolerance of time difference (s)
MAXDTOE    = 7200.0

# ---------------------------------------------------------------------------
# Positioning modes  (PMODE_*)
# ---------------------------------------------------------------------------
PMODE_SINGLE       = 0
PMODE_TDCP         = 1
PMODE_DGPS         = 2
PMODE_KINEMA       = 3
PMODE_STATIC       = 4
PMODE_STATIC_START = 5
PMODE_MOVEB        = 6
PMODE_FIXED        = 7
PMODE_PPP_KINEMA   = 8
PMODE_PPP_STATIC   = 9
PMODE_PPP_FIXED    = 10
PMODE_INS_MECH     = 11
PMODE_LC_POS       = 12
PMODE_LC_SPP       = 13
PMODE_LC_DGPS      = 14
PMODE_LC_PPK       = 15
PMODE_LC_PPP       = 16
PMODE_TC_SPP       = 17
PMODE_TC_TDCP      = 18
PMODE_TC_DGPS      = 19
PMODE_TC_PPK       = 20
PMODE_TC_PPP       = 21
PMODE_STC_PPK      = 22
PMODE_STC_PPP      = 23

# Human-readable mode strings (index == PMODE_*)
KPMODESTR = [
    "SPP", "TDCP", "DGPS", "PPK-KINE", "PPK-STATIC",
    "PPK-S-START", "PPK-MOVEB", "PPK-FIXED",
    "PPP-KINE", "PPP-STATIC", "PPP-FIXED",
]

# ---------------------------------------------------------------------------
# Ambiguity resolution modes  (ARMODE_*)
# ---------------------------------------------------------------------------
ARMODE_OFF      = 0
ARMODE_CONT     = 1
ARMODE_INST     = 2
ARMODE_FIXHOLD  = 3
ARMODE_PPPAR    = 4
ARMODE_PPPAR_ILS = 5
ARMODE_WLNL    = 6
ARMODE_TCAR    = 7

# Ambiguity resolution product types
AR_PROD_IRC     = 1
AR_PROD_FCB     = 2
AR_PROD_UPD     = 3
AR_PROD_OSB_GRM = 4
AR_PROD_OSB_WHU = 5
AR_PROD_OSB_COM = 6
AR_PROD_OSB_SGG = 7
AR_PROD_OSB_CNT = 8

# ---------------------------------------------------------------------------
# Ionosphere options  (IONOOPT_*)
# ---------------------------------------------------------------------------
IONOOPT_OFF     = 0
IONOOPT_BRDC    = 1
IONOOPT_SBAS    = 2
IONOOPT_IFLC    = 3   # L1/L2 ionosphere-free LC
IONOOPT_EST     = 4
IONOOPT_TEC     = 5
IONOOPT_QZS     = 6
IONOOPT_UC      = 7   # uncombined
IONOOPT_UC_CONS = 8   # uncombined + constrained
IONOOPT_IF2     = 9

# Troposphere options  (TROPOPT_*)
TROPOPT_OFF  = 0
TROPOPT_SAAS = 1
TROPOPT_SBAS = 2
TROPOPT_EST  = 3
TROPOPT_ESTG = 4
TROPOPT_ZTD  = 5

# Satellite ephemeris  (EPHOPT_*)
EPHOPT_BRDC  = 0
EPHOPT_PREC  = 1
EPHOPT_SBAS  = 2
EPHOPT_SSRAPC = 3
EPHOPT_SSRCOM = 4
EPHOPT_LEXAPC = 5
EPHOPT_LEXCOM = 6

# Code bias options  (CBIAS_OPT_*)
CBIAS_OPT_OFF     = 0
CBIAS_OPT_BRD_TGD = 1
CBIAS_OPT_COD_DCB = 2
CBIAS_OPT_IGG_DCB = 3
CBIAS_OPT_GBM_DCB = 4
CBIAS_OPT_MIX_DCB = 5
CBIAS_OPT_OSB     = 6

# BDS-3 options  (BD3OPT_*)
BD3OPT_OFF   = 0
BD3OPT_BD2_3 = 1
BD3OPT_BD23  = 2
BD3OPT_BD3   = 3

# GLO AR modes  (GLO_ARMODE_*)
GLO_ARMODE_OFF     = 0
GLO_ARMODE_FIXHOLD = 1
GLO_ARMODE_AUTOCAL = 2
GLO_ARMODE_AUTOCALC = 3

# Dynamics modes
DYNAMICS_OFF = 0
DYNAMICS_ON  = 1

# INS alignment modes  (INS_ALIGN_*)
INS_ALIGN_GNSS_PV   = 0
INS_ALIGN_GNSS_PPK  = 1
INS_ALIGN_GNSS_DGPS = 2

# ---------------------------------------------------------------------------
# Solution quality / format flags
# ---------------------------------------------------------------------------
SOLQ_NONE  = 0
SOLQ_FIX   = 1
SOLQ_FLOAT = 2
SOLQ_SBAS  = 3
SOLQ_DGPS  = 4
SOLQ_SINGLE = 5
SOLQ_PPP   = 6
SOLQ_DR    = 7

SOLF_LLH    = 0
SOLF_XYZ    = 1
SOLF_ENU    = 2
SOLF_NMEA   = 3
SOLF_STAT   = 4
SOLF_GSIF   = 5
SOLF_INS_LLH = 6
SOLF_INS_XYZ = 7
SOLF_INS_YGM = 8

# ---------------------------------------------------------------------------
# Kalman filter type flags  (KFOPT_*)
# ---------------------------------------------------------------------------
KFOPT_STD      = 0   # standard KF  (Joseph form)
KFOPT_VBKF     = 1   # Variational Bayes adaptive KF
KFOPT_SAGE_HUSA = 2  # Sage-Husa adaptive KF

# Robust QC options  (ROBUST_QC_*)
ROBUST_QC_OFF    = 0
ROBUST_QC_IGG_PR = 1
ROBUST_QC_IGG_CP = 2
ROBUST_QC_IGG    = 3
ROBUST_QC_SHI    = 4
ROBUST_QC_ZHAO   = 5

# ---------------------------------------------------------------------------
# Error factors by system
# ---------------------------------------------------------------------------
EFACT_GPS = 1.0
EFACT_GLO = 2.0
EFACT_GAL = 1.0
EFACT_QZS = 1.0
EFACT_CMP = 1.0
EFACT_IRN = 1.5
EFACT_SBS = 3.0

# ---------------------------------------------------------------------------
# PPP-specific thresholds / initial variances (from ppp.c)
# ---------------------------------------------------------------------------
MAX_ITER      = 8
MAX_STD_FIX   = 0.15
MIN_NSAT_SOL  = 4
THRES_REJECT  = 4.0
THRES_MW_JUMP = 10.0

VAR_POS  = 60.0 ** 2
VAR_VEL  = 60.0 ** 2
VAR_ACC  = 60.0 ** 2
VAR_CLK  = 60.0 ** 2
VAR_DCB  = 60.0 ** 2
VAR_IFCB = 60.0 ** 2
VAR_ZWD  = 0.03 ** 2
VAR_GRA  = 0.01 ** 2
VAR_BIAS = 60.0 ** 2
VAR_IONO = 60.0 ** 2
VAR_GLO_IFB = 0.6 ** 2

ERR_SAAS  = 0.3
ERR_BRDCI = 0.5
ERR_CBIAS = 0.3
REL_HUMI  = 0.7
GAP_RESION = 120

EFACT_GPS_L5 = 10.0

MUDOT_GPS   = 0.00836 * D2R
MUDOT_GLO   = 0.00888 * D2R
EPS0_GPS    = 13.5 * D2R
EPS0_GLO    = 14.2 * D2R
T_POSTSHADOW = 1800.0
QZS_EC_BETA = 20.0

# LAMBDA / ILS AR
MIN_AMB_RES = 4
MIN_LOCK_AR = 15
LOG_PI  = 1.14472988584940017
SQRT2   = 1.41421356237309510

# ---------------------------------------------------------------------------
# Student-t distribution quantiles used in robust residual QC
# (from the C++ source; index = degrees of freedom)
# tdistb_0250[n]: two-sided 25 % point  (p = 0.75)
# tdistb_0005[n]: two-sided  0.5% point (p = 0.995)
# tdistb_0050[n]: two-sided  5 % point  (p = 0.95)
# tdistb_0010[n]: two-sided  1 % point  (p = 0.99)
# ---------------------------------------------------------------------------
TDISTB_0250 = [
    0.0, 1.000, 0.816, 0.765, 0.741, 0.727,
    0.718, 0.711, 0.706, 0.703, 0.700,
    0.697, 0.695, 0.694, 0.692, 0.691,
    0.690, 0.689, 0.688, 0.688, 0.687,
    0.686, 0.686, 0.685, 0.685, 0.684,
    0.684, 0.684, 0.683, 0.683, 0.683,
]

TDISTB_0005 = [
    0.0, 63.657, 9.925, 5.841, 4.604, 4.032,
    3.707, 3.499, 3.355, 3.250, 3.169,
    3.106, 3.055, 3.012, 2.977, 2.947,
    2.921, 2.898, 2.878, 2.861, 2.845,
    2.831, 2.819, 2.807, 2.797, 2.787,
    2.779, 2.771, 2.763, 2.756, 2.750,
]

TDISTB_0050 = [
    0.0, 6.314, 2.920, 2.353, 2.132, 2.015,
    1.943, 1.895, 1.860, 1.833, 1.812,
    1.796, 1.782, 1.771, 1.761, 1.753,
    1.746, 1.740, 1.734, 1.729, 1.725,
    1.721, 1.717, 1.714, 1.711, 1.708,
    1.706, 1.703, 1.701, 1.699, 1.697,
]

TDISTB_0010 = [
    0.0, 31.821, 6.965, 4.541, 3.747, 3.365,
    3.143, 2.998, 2.896, 2.821, 2.764,
    2.718, 2.681, 2.650, 2.624, 2.602,
    2.583, 2.567, 2.552, 2.539, 2.528,
    2.518, 2.508, 2.500, 2.492, 2.485,
    2.479, 2.473, 2.467, 2.462, 2.457,
]