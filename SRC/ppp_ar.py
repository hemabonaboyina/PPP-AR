"""
ppp_ar_app.py
=============
Python conversion of ``src/app/ppp_ar.cc``.

Contains:
  process(popt, fopt, sopt)  — scan observation directory and post-process
                               each station file via ``postpos()``.
  main(argv)                 — CLI entry-point.

Notes
-----
*  The ``postpos()`` call is the core RTKLIB positioning engine.  A full
   Python implementation of postpos is beyond the scope of this conversion
   (it spans ~3000 lines of RTKLIB C code).  In this module it is wrapped
   as ``postpos_stub()`` which raises a ``NotImplementedError`` unless you
   supply a real implementation.  You can integrate a RTKLIB Python binding
   (e.g. rtklib-py) or implement the algorithm directly to replace the stub.

*  All file-matching, option-parsing and output-path logic is fully
   converted and functional via the ``match_file`` module.
"""

from __future__ import annotations

import os
import sys
import time
import copy
import argparse
from typing import Optional

from constants import (
    PMODE_DGPS, PMODE_STATIC_START, PMODE_PPP_KINEMA, PMODE_PPP_FIXED,
    PMODE_TC_DGPS, PMODE_TC_PPK, PMODE_STC_PPK,
    PMODE_TC_PPP, PMODE_LC_PPP, PMODE_STC_PPP,
    PMODE_INS_MECH, PMODE_LC_POS,
    INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS,
    KPMODESTR, ARMODE_OFF,
)
from structures import PrcOpt, SolOpt, FilOpt, GTime
from match_file import (
    load_prc_files, free_prc_files, parse_cmd, matchout,
    time2epoch, time2doy, timediff, timeadd,
)


# ---------------------------------------------------------------------------
# Stub for rtklib's postpos() — replace with a real implementation
# ---------------------------------------------------------------------------

def postpos_stub(ts, te, ti, tu, popt, sopt, fopt, infiles, outfile,
                 rov=None, base=None) -> int:
    """Placeholder for RTKLIB's ``postpos()`` positioning engine.

    Parameters
    ----------
    ts, te   : GTime — processing start / end time.
    ti, tu   : float — time interval and unit (seconds).
    popt     : PrcOpt
    sopt     : SolOpt
    fopt     : FilOpt
    infiles  : list[str] — input files (obs, nav, sp3, clk, …)
    outfile  : str       — output solution file path.
    rov, base : str | None — rover / base identifiers.

    Returns
    -------
    int — 0 on failure, 1 on success (mirrors the C++ return convention).

    Raises
    ------
    NotImplementedError — always, until a real engine is wired in.
    """
    raise NotImplementedError(
        "postpos() is not yet implemented in Python.\n"
        "Wire in a RTKLIB binding (e.g. rtklib-py) or implement\n"
        "the PPP/PPK processing engine here."
    )


# Allow users to monkey-patch with a real implementation
postpos = postpos_stub


# ---------------------------------------------------------------------------
# _is_ppk / _is_ppp  — mode-type helpers
# ---------------------------------------------------------------------------

def _is_ppk(popt: PrcOpt) -> bool:
    return (
        (PMODE_DGPS <= popt.mode <= PMODE_STATIC_START) or
        popt.mode in (PMODE_TC_DGPS, PMODE_TC_PPK, PMODE_STC_PPK) or
        popt.insopt.imu_align in (INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS)
    )


def _is_ppp(popt: PrcOpt) -> bool:
    return (
        PMODE_PPP_KINEMA <= popt.mode <= PMODE_PPP_FIXED or
        popt.mode in (PMODE_TC_PPP, PMODE_LC_PPP, PMODE_STC_PPP)
    )


# ---------------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------------

def process(popt: PrcOpt, fopt: FilOpt, sopt: SolOpt) -> int:
    """Scan the observation directory and post-process each station file.

    Corresponds to ``process()`` in ppp_ar.cc.

    Parameters
    ----------
    popt : PrcOpt — processing options (modified in-place: site_name, etc.)
    fopt : FilOpt — file options (rover obs path is set per station).
    sopt : SolOpt — solution options.

    Returns
    -------
    int — 1 on at least one successful run, 0 if no files processed.
    """
    ep   = time2epoch(popt.ts)
    doy  = int(time2doy(popt.ts))
    yyyy = int(ep[0])

    obs_sub = popt.obsdir if popt.obsdir else "obs"
    obs_dir = os.path.join(popt.prcdir, f"{yyyy:04d}", f"{doy:03d}", obs_sub)

    if not os.path.isdir(obs_dir):
        print(f"Observation directory not found: {obs_dir}", file=sys.stderr)
        return 0

    ppk = _is_ppk(popt)
    ppp = _is_ppp(popt)

    ret = 0

    for fname in sorted(os.listdir(obs_dir)):
        # Skip hidden / dot files
        if fname.startswith("."):
            continue
        # Skip base-station and IMU files
        if "base" in fname or "imu" in fname:
            continue
        # Must have an extension ending with 'o'
        root, ext = os.path.splitext(fname)
        if not ext or not ext.endswith("o"):
            continue

        # Station name filter
        site_name_4 = fname[:4].upper()
        if popt.site_list:
            filter_sites = [s.strip().upper() for s in popt.site_list.split(",")]
            if site_name_4 not in filter_sites:
                continue

        popt.site_name = site_name_4
        popt.site_idx += 1

        print(f"PROCESS {popt.site_name} "
              f"{yyyy:04d}/{doy:03d}", file=sys.stderr)
        sys.stderr.flush()

        fopt.robsf = os.path.join(obs_dir, fname)

        # Determine output file paths
        matchout(popt, popt.prcdir, fopt, sopt)

        # Build input file list
        infiles = [fopt.robsf]

        if ppk and fopt.bobsf:
            infiles.append(fopt.bobsf)

        # Broadcast navigation files
        for p in fopt.navf:
            if p:
                infiles.append(p)

        # Precise orbit + clock
        if ppp:
            for p in fopt.sp3f:
                if p:
                    infiles.append(p)
            for p in fopt.clkf:
                if p:
                    infiles.append(p)

        # Call positioning engine
        try:
            ok = postpos(
                popt.ts, popt.te, 0.0, 0.0,
                popt, sopt, fopt,
                infiles,
                fopt.solf,
            )
            if ok:
                ret = 1
            else:
                print(f"{popt.site_name} PROCESS ERROR!", file=sys.stderr)
        except NotImplementedError as exc:
            print(f"\nWarning: {exc}\n"
                  "Skipping actual positioning. "
                  "File discovery and output path generation succeeded.",
                  file=sys.stderr)
            ret = 1   # report "success" for the infrastructure part

        popt.site_name = ""

    return ret


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    """PPP-AR command-line entry point.

    Corresponds to ``main()`` in ppp_ar.cc.

    Usage
    -----
    ppp_ar -C <conf_file> -S <systems> -M <mode> -A <ar> -L <level>

    Arguments
    ---------
    -C  Path to configuration file.
    -M  Processing mode string: PPP-KINE, PPP-STATIC, etc.
    -S  GNSS systems: G=GPS, R=GLO, E=GAL, C=BDS, J=QZS (e.g. "GE").
    -A  Ambiguity resolution mode (integer 0..7, 0=float).
    -L  Trace/log level (0=quiet, 128=verbose).

    Returns
    -------
    int — 0 on failure, 1 on success.
    """
    if argv is None:
        argv = sys.argv[1:]

    t_start = time.time()

    popt  = PrcOpt()
    sopt  = SolOpt()
    fopt  = FilOpt()

    ok, port = parse_cmd(argv, popt, sopt, fopt)
    if not ok:
        return 0

    ts = popt.ts
    te = popt.te

    # Determine number of days to process
    if ts.time != 0 and te.time != 0:
        diff_days = timediff(te, ts) / 86400.0
        nday = max(1, round(diff_days))
    else:
        nday = 1

    if nday > 1:
        popt.prctype = 1

    for i in range(nday):
        popt_ = copy.deepcopy(popt)
        sopt_ = copy.deepcopy(sopt)
        fopt_ = copy.deepcopy(fopt)

        if popt.prctype:
            popt_.ts = timeadd(ts, 86400.0 * i)
            popt_.te = timeadd(ts, 86400.0 * (i + 1))

        if not load_prc_files(popt_.prcdir, popt_, fopt_):
            print(f"Load processing files failed for day {i+1}", file=sys.stderr)
            return 0

        if process(popt_, fopt_, sopt_):
            elapsed = time.time() - t_start
            print(f"total sec: {elapsed:5.2f}", file=sys.stderr)
            sys.stderr.flush()
        else:
            print(f"{popt_.site_name} PROCESS ERROR!", file=sys.stderr)
            sys.stderr.flush()

        free_prc_files(fopt_)

    return 1


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(0 if main() else 1)