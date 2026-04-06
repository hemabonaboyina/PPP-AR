"""
PPP-AR: Multi-GNSS Precise Point Positioning with Ambiguity Resolution
Python conversion of PPP_AR-master C++ project (based on RTKLIB).

Modules:
  constants   - RTKLIB-derived constants, enums, and GNSS system flags
  structures  - Core data structures (GTime, PrcOpt, SolOpt, FilOpt, etc.)
  kf          - Kalman filter variants (standard, VBAKF, Sage-Husa)
  qc          - Quality control / residual checking (IGG, SHI, ZHAO)
  match_file  - File-matching utilities (nav, sp3, clk, bia, atx, …)
  ppp_ar_app  - Main application entry point (process + CLI)
"""

from .constants import *          # noqa: F401,F403
from .structures import *         # noqa: F401,F403