"""
match_file.py
=============
Python conversion of ``src/LibUnit/match_file.cc``.

Provides utilities for locating and matching GNSS ancillary product files
(navigation, precise orbit/clock, bias, EOP, ATX, etc.) according to the
processing options and epoch.

The RTKLIB time functions referenced in the C++ code
(``time2epoch``, ``time2doy``, ``time2gpst``, ``timeadd``, ``epoch2time``,
``timediff``) are re-implemented here using Python's ``datetime`` module.

Public API
----------
load_prc_files(prc_dir, popt, fopt)  → bool
    Load all ancillary product file paths into *fopt*.

free_prc_files(fopt)
    Reset file-path strings to empty.

load_conf(conf_file, popt, sopt, fopt) → bool
    Load a configuration file (thin wrapper — real option parsing
    would call into rtklib's ``loadopts``; here we parse the key
    ``-C / -M / -S / -A / -L`` arguments).

parse_cmd(argv, popt, sopt, fopt)  → (bool, int)
    Parse the PPP-AR command-line argument list.

match_nav_file(popt, ts, prc_dir)   → list[str]
    Return (up to 3) navigation-file paths for the triplet day.

match_clk_file(popt, ts, prc_dir)   → list[str]
    Return (up to 3) precise-clock file paths.

match_sp3_file(popt, ts, prc_dir)   → list[str]
    Return (up to 3) precise-orbit (SP3) file paths.

match_bia_file(opt, ts, prc_dir)    → str
    Return the OSB/bias file path.

match_fcb_file(opt, ts, prc_dir)    → str
    Return the FCB file path.

match_upd_file(opt, ts, prc_dir)    → list[str]
    Return (up to 3) UPD file paths (EWL, WL, NL).

match_eop_file(ts, prc_dir, popt)   → str
    Return the ERP/EOP file path.

match_dcb_file(popt, ts, prc_dir)   → str
    Return the DCB file glob pattern.

match_mgexdcb_file(ts, prc_dir, opt) → str
    Return the MGEX DCB/BSX file path.

match_ion_file(ts, prc_dir)         → str
    Return the ionosphere (IONEX) file path.

match_atx_file(ts, prc_dir, popt)   → str
    Return the antenna ATX file path.

match_blq_file(ts, prc_dir)         → str
    Return the BLQ ocean loading file path.

match_base_obs_file(opt, ts, prc_dir) → str
    Return the base-station observation file path.

matchout(popt, prc_dir, fopt, sopt)
    Determine output solution file paths and write them into *fopt*.
"""

from __future__ import annotations

import os
import sys
import datetime
from dataclasses import replace
from typing import Optional, Tuple

from constants import (
    PMODE_DGPS, PMODE_STATIC_START, PMODE_PPP_KINEMA, PMODE_PPP_FIXED,
    PMODE_TC_DGPS, PMODE_TC_PPK, PMODE_LC_DGPS, PMODE_LC_PPK,
    PMODE_TC_PPP, PMODE_LC_PPP, PMODE_INS_MECH, PMODE_LC_POS,
    PMODE_STC_PPP, PMODE_STC_PPK,
    EPHOPT_PREC,
    CBIAS_OPT_BRD_TGD, CBIAS_OPT_COD_DCB,
    CBIAS_OPT_IGG_DCB, CBIAS_OPT_GBM_DCB, CBIAS_OPT_MIX_DCB,
    IONOOPT_TEC, IONOOPT_UC_CONS,
    ARMODE_PPPAR, ARMODE_PPPAR_ILS, ARMODE_OFF, ARMODE_CONT,
    ARMODE_INST, ARMODE_FIXHOLD,
    AR_PROD_FCB, AR_PROD_OSB_WHU, AR_PROD_OSB_GRM, AR_PROD_OSB_COM,
    AR_PROD_OSB_SGG, AR_PROD_OSB_CNT, AR_PROD_UPD, AR_PROD_IRC,
    BD3OPT_OFF, BD3OPT_BD2_3, BD3OPT_BD23, BD3OPT_BD3,
    SYS_GPS, SYS_GLO, SYS_GAL, SYS_CMP, SYS_BD3, SYS_QZS, SYS_NONE,
    INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS,
    KPMODESTR, ARMODE_OFF,
)
from structures import PrcOpt, SolOpt, FilOpt, GTime

# ---------------------------------------------------------------------------
# GPS epoch: 1980-01-06 00:00:00 UTC
# ---------------------------------------------------------------------------
_GPS_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0)
_UNIX_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0)

# Leap seconds table (GPS - UTC), approximate list through 2025
_LEAP_SECONDS = [
    (datetime.datetime(1981, 7, 1), 1),
    (datetime.datetime(1982, 7, 1), 2),
    (datetime.datetime(1983, 7, 1), 3),
    (datetime.datetime(1985, 7, 1), 4),
    (datetime.datetime(1988, 1, 1), 5),
    (datetime.datetime(1990, 1, 1), 6),
    (datetime.datetime(1991, 1, 1), 7),
    (datetime.datetime(1992, 7, 1), 8),
    (datetime.datetime(1993, 7, 1), 9),
    (datetime.datetime(1994, 7, 1), 10),
    (datetime.datetime(1996, 1, 1), 11),
    (datetime.datetime(1997, 7, 1), 12),
    (datetime.datetime(1999, 1, 1), 13),
    (datetime.datetime(2006, 1, 1), 14),
    (datetime.datetime(2009, 1, 1), 15),
    (datetime.datetime(2012, 7, 1), 16),
    (datetime.datetime(2015, 7, 1), 17),
    (datetime.datetime(2017, 1, 1), 18),
]

# ---------------------------------------------------------------------------
# RTKLIB time conversion helpers
# ---------------------------------------------------------------------------

def gtime_to_datetime(t: GTime) -> datetime.datetime:
    """Convert a GTime to a Python datetime (UTC)."""
    dt = _UNIX_EPOCH + datetime.timedelta(seconds=float(t.time) + t.sec)
    return dt


def datetime_to_gtime(dt: datetime.datetime) -> GTime:
    """Convert a Python datetime (UTC) to a GTime."""
    delta = dt - _UNIX_EPOCH
    total = delta.total_seconds()
    time_ = int(total)
    sec_  = total - time_
    return GTime(time=time_, sec=sec_)


def time2epoch(t: GTime) -> list:
    """Convert GTime to [year, month, day, hour, min, sec] list."""
    dt = gtime_to_datetime(t)
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute,
            dt.second + dt.microsecond * 1e-6]


def epoch2time(ep: list) -> GTime:
    """Convert [year, month, day, h, m, s] to GTime."""
    dt = datetime.datetime(int(ep[0]), int(ep[1]), int(ep[2]),
                           int(ep[3]), int(ep[4]),
                           int(ep[5]), int((ep[5] % 1) * 1e6))
    return datetime_to_gtime(dt)


def time2doy(t: GTime) -> float:
    """Return fractional day-of-year for a GTime."""
    dt = gtime_to_datetime(t)
    jan1 = datetime.datetime(dt.year, 1, 1)
    return (dt - jan1).total_seconds() / 86400.0 + 1.0


def _leaps_at(dt: datetime.datetime) -> int:
    """Return cumulative GPS-UTC leap seconds at the given UTC datetime."""
    ls = 0
    for boundary, n in _LEAP_SECONDS:
        if dt >= boundary:
            ls = n
    return ls


def time2gpst(t: GTime) -> Tuple[float, int]:
    """Return (tow_seconds, gps_week) for a GTime.

    Returns
    -------
    (tow, week) where tow is seconds-of-week (0..604800).
    """
    dt = gtime_to_datetime(t)
    ls = _leaps_at(dt)
    # GPS time = UTC + leap seconds
    gps_dt = dt + datetime.timedelta(seconds=ls)
    delta  = gps_dt - _GPS_EPOCH
    total  = delta.total_seconds()
    week   = int(total // 604800)
    tow    = total - week * 604800.0
    return tow, week


def timeadd(t: GTime, sec: float) -> GTime:
    """Add *sec* seconds to a GTime."""
    total = float(t.time) + t.sec + sec
    ti    = int(total)
    ts_   = total - ti
    return GTime(time=ti, sec=ts_)


def timediff(t1: GTime, t2: GTime) -> float:
    """Return t1 - t2 in seconds."""
    return (float(t1.time) - float(t2.time)) + (t1.sec - t2.sec)


# ---------------------------------------------------------------------------
# Analysis-centre name helpers
# ---------------------------------------------------------------------------

def _getupac(ac: str) -> str:
    """Return 3-char upper-case AC code for long filename format."""
    _map = {"com": "COD", "wum": "WUM", "grm": "GRG",
            "gfz": "GFZ", "gbm": "GBM", "cnt": "CNT"}
    return _map.get(ac.lower(), ac[:3].upper())


# ---------------------------------------------------------------------------
# match_nav_file
# ---------------------------------------------------------------------------

def match_nav_file(popt: PrcOpt, ts: GTime, prc_dir: str) -> list:
    """Locate navigation files for the day triplet (prev, curr, next).

    Corresponds to ``match_navfile()`` in match_file.cc.

    Returns
    -------
    list[str] — up to 3 paths (empty string if not found).
    """
    paths = ["", "", ""]
    sep   = os.sep

    for i, dt_offset in enumerate([-86400.0, 0.0, 86400.0]):
        t    = timeadd(ts, dt_offset)
        ep   = time2epoch(t)
        yyyy = int(ep[0])
        doy  = int(time2doy(t))
        yy   = yyyy - 2000 if yyyy >= 2000 else yyyy - 1900
        d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                            "products", "nav")

        for fmt in [
            f"brdm{doy:03d}0.{yy:02d}p",
            f"brdc{doy:03d}0.{yy:02d}n",
            f"BRDM00DLR_S_{yyyy:04d}{doy:03d}0000_01D_MN.rnx",
            f"BRDC00IGN_R_{yyyy:04d}{doy:03d}0000_01D_MN.rnx",
        ]:
            candidate = os.path.join(d, fmt)
            if os.path.isfile(candidate):
                paths[i] = candidate
                break

    return paths


# ---------------------------------------------------------------------------
# match_clk_file
# ---------------------------------------------------------------------------

def match_clk_file(popt: PrcOpt, ts: GTime, prc_dir: str) -> list:
    """Locate precise clock files for the day triplet.

    Corresponds to ``match_clkfile()`` in match_file.cc.

    Returns
    -------
    list[str] — up to 3 paths.  Empty string on index 1 means failure.
    """
    paths = ["", "", ""]
    sep   = os.sep
    upac  = _getupac(popt.ac_name)

    for i, dt_offset in enumerate([-86400.0, 0.0, 86400.0]):
        t    = timeadd(ts, dt_offset)
        ep   = time2epoch(t)
        yyyy = int(ep[0])
        doy  = int(time2doy(t))
        tow, week = time2gpst(t)
        dow  = int(tow / 86400.0)
        d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                            "products", popt.ac_name)

        if popt.aclong:
            if (popt.modear == ARMODE_PPPAR_ILS and
                    popt.arprod == AR_PROD_OSB_WHU and
                    popt.ac_name.lower() == "wum"):
                fname = f"WHU5MGXFIN_{yyyy:04d}{doy:03d}0000_01D_30S_CLK.CLK"
            else:
                fname = (f"{upac}0MGX{popt.prdtype}_{yyyy:04d}{doy:03d}0000"
                         f"_01D_{popt.clk_int}_CLK.CLK")
        else:
            if (popt.modear == ARMODE_PPPAR_ILS and
                    popt.arprod == AR_PROD_OSB_WHU and
                    popt.ac_name.lower() == "wum"):
                fname = f"whu{week:04d}{dow}.clk"
            else:
                fname = f"{popt.ac_name}{week:04d}{dow}.clk"

        candidate = os.path.join(d, fname)
        if os.path.isfile(candidate):
            paths[i] = candidate
        elif i == 1:
            print(f"Miss precise clock file: {candidate}", file=sys.stderr)

    return paths


# ---------------------------------------------------------------------------
# match_sp3_file
# ---------------------------------------------------------------------------

def match_sp3_file(popt: PrcOpt, ts: GTime, prc_dir: str) -> list:
    """Locate precise orbit (SP3) files for the day triplet.

    Corresponds to ``match_sp3file()`` in match_file.cc.

    Returns
    -------
    list[str] — up to 3 paths.
    """
    paths = ["", "", ""]
    upac  = _getupac(popt.ac_name)

    for i, dt_offset in enumerate([-86400.0, 0.0, 86400.0]):
        t    = timeadd(ts, dt_offset)
        ep   = time2epoch(t)
        yyyy = int(ep[0])
        doy  = int(time2doy(t))
        tow, week = time2gpst(t)
        dow  = int(tow / 86400.0)
        d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                            "products", popt.ac_name)

        if popt.aclong:
            fname = (f"{upac}0MGX{popt.prdtype}_{yyyy:04d}{doy:03d}0000"
                     f"_01D_{popt.eph_int}_ORB.SP3")
        else:
            fname = f"{popt.ac_name}{week:04d}{dow}.sp3"

        candidate = os.path.join(d, fname)
        if os.path.isfile(candidate):
            paths[i] = candidate
        elif i == 1:
            print(f"Miss precise orbit: {candidate}", file=sys.stderr)

    return paths


# ---------------------------------------------------------------------------
# match_eop_file
# ---------------------------------------------------------------------------

def match_eop_file(ts: GTime, prc_dir: str, popt: PrcOpt) -> str:
    """Locate EOP/ERP file.

    Corresponds to ``match_eopfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    doy  = int(time2doy(ts))
    tow, week = time2gpst(ts)

    d = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                     "products", popt.ac_name)

    if popt.aclong:
        if popt.ac_name.lower() == "com":
            fname = f"COD0MGXFIN_{yyyy:04d}{doy:03d}0000_03D_12H_ERP.ERP"
        elif popt.ac_name.lower() == "gbm":
            fname = f"GBM0MGXRAP_{yyyy:04d}{doy:03d}0000_01D_01D_ERP.ERP"
        else:
            upac  = _getupac(popt.ac_name)
            fname = f"{upac}0MGXFIN_{yyyy:04d}{doy:03d}0000_01D_01D_ERP.ERP"
    else:
        fname = f"{popt.ac_name}{week:04d}7.erp"

    candidate = os.path.join(d, fname)
    if os.path.isfile(candidate):
        return candidate

    # Fallback: IGS weekly ERP
    fallback = os.path.join(prc_dir, f"{yyyy:04d}", "igs_erp",
                            f"igs{week:04d}7.erp")
    if os.path.isfile(fallback):
        return fallback

    print(f"Miss erp file: {candidate}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# match_dcb_file
# ---------------------------------------------------------------------------

def match_dcb_file(popt: PrcOpt, ts: GTime, prc_dir: str) -> str:
    """Return DCB file glob pattern.

    Corresponds to ``match_dcbfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    mon  = int(ep[1])
    if mon == 0:
        return ""
    d = os.path.join(prc_dir, f"{yyyy:04d}", "dcb")
    return os.path.join(d, f"*{mon:02d}*.DCB")


# ---------------------------------------------------------------------------
# match_ion_file
# ---------------------------------------------------------------------------

def match_ion_file(ts: GTime, prc_dir: str) -> str:
    """Locate IONEX file.

    Corresponds to ``match_ionfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    doy  = int(time2doy(ts))
    yy   = yyyy - 2000 if yyyy >= 2000 else yyyy - 1900
    d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}products",
                        "igs_ion")
    candidate = os.path.join(d, f"CODG{doy:03d}0.{yy:02d}I")
    if os.path.isfile(candidate):
        return candidate
    return ""


# ---------------------------------------------------------------------------
# match_fcb_file
# ---------------------------------------------------------------------------

def match_fcb_file(opt: PrcOpt, ts: GTime, prc_dir: str) -> str:
    """Locate FCB file.

    Corresponds to ``match_fcbfile()`` in match_file.cc.
    """
    tow, week = time2gpst(ts)
    wod  = int(tow / 86400.0)
    doy  = int(time2doy(ts))
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    upac = _getupac(opt.ac_name)
    d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                        "products", "fcb")
    fname = f"sgg{week:04d}{wod}_{upac}0MGX{opt.prdtype}.fcb"
    candidate = os.path.join(d, fname)
    if os.path.isfile(candidate):
        return candidate
    print(f"Miss fcb file: {candidate}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# match_bia_file
# ---------------------------------------------------------------------------

def match_bia_file(opt: PrcOpt, ts: GTime, prc_dir: str) -> str:
    """Locate OSB / bias file.

    Corresponds to ``match_biafile()`` in match_file.cc.
    """
    tow, week = time2gpst(ts)
    wod  = int(tow / 86400.0)
    doy  = int(time2doy(ts))
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}", "products")

    fname = ""
    if opt.arprod == AR_PROD_OSB_WHU and opt.ac_name.lower() == "wum":
        if opt.aclong:
            fname = f"WHU0MGXFIN_{yyyy:04d}{doy:03d}0000_01D_01D_ABS.BIA"
        else:
            fname = f"whu{year:04d}{wod}.bia"
        candidate = os.path.join(d, opt.ac_name, fname)
    elif opt.arprod == AR_PROD_OSB_CNT and opt.ac_name.lower() == "cnt":
        fname     = f"cnt{week:04d}{wod}.bia"
        candidate = os.path.join(d, opt.ac_name, fname)
    elif opt.arprod == AR_PROD_OSB_GRM and opt.ac_name.lower() == "gbm":
        fname     = f"gbm{week:04d}{wod}.bia"
        candidate = os.path.join(d, opt.ac_name, fname)
    elif opt.arprod == AR_PROD_OSB_COM and opt.ac_name.lower() == "com":
        if opt.aclong:
            fname = f"COD0MGXFIN_{yyyy:04d}{doy:03d}0000_01D_01D_OSB.BIA"
        else:
            fname = f"com{week:04d}{wod}.bia"
        candidate = os.path.join(d, opt.ac_name, fname)
    elif opt.arprod == AR_PROD_OSB_SGG and opt.ac_name.lower() == "com":
        fname     = f"SGG{week:04d}{wod}.BIA"
        candidate = os.path.join(d, opt.ac_name, fname)
    else:
        print(f"Unknown arprod/ac_name combination: {opt.arprod}/{opt.ac_name}",
              file=sys.stderr)
        return ""

    if os.path.isfile(candidate):
        return candidate
    print(f"Miss bias file: {candidate}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# match_upd_file
# ---------------------------------------------------------------------------

def match_upd_file(opt: PrcOpt, ts: GTime, prc_dir: str) -> list:
    """Locate UPD files (EWL, WL, NL).

    Corresponds to ``match_updfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    doy  = int(time2doy(ts))
    yyyy = int(ep[0])
    d    = os.path.join(prc_dir, "products", "upd")

    paths = ["", "", ""]
    for i, tag in enumerate([f"upd_ewl_{yyyy:04d}{doy:03d}_GEC",
                              f"upd_wl_{yyyy:04d}{doy:03d}_GREC",
                              f"upd_nl_{yyyy:04d}{doy:03d}_GREC"]):
        candidate = os.path.join(d, tag)
        if os.path.isfile(candidate):
            paths[i] = candidate
        elif i >= 1:
            print(f"Miss upd file: {candidate}", file=sys.stderr)
    return paths


# ---------------------------------------------------------------------------
# match_mgexdcb_file
# ---------------------------------------------------------------------------

def match_mgexdcb_file(ts: GTime, prc_dir: str, opt_int: int) -> str:
    """Locate MGEX DCB/BSX file.

    Corresponds to ``match_mgexdcbfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    doy  = int(time2doy(ts))

    if opt_int in (CBIAS_OPT_IGG_DCB, CBIAS_OPT_MIX_DCB):
        d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                            "products", "cas")
        fname = f"CAS0MGXRAP_{yyyy:04d}{doy:03d}0000_01D_01D_DCB.BSX"
    elif opt_int == CBIAS_OPT_GBM_DCB:
        d    = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                            "products", "gbm")
        fname = f"GBM0MGXRAP_{yyyy:04d}{doy:03d}0000_01D_01D_REL.BIA"
    else:
        return ""

    candidate = os.path.join(d, fname)
    if os.path.isfile(candidate):
        return candidate
    return ""


# ---------------------------------------------------------------------------
# match_blq_file
# ---------------------------------------------------------------------------

def match_blq_file(prc_dir: str) -> str:
    """Locate BLQ ocean loading file.

    Corresponds to ``match_blqfile()`` in match_file.cc.
    """
    candidate = os.path.join(prc_dir, "blq", "ocnload.blq")
    return candidate if os.path.isfile(candidate) else ""


# ---------------------------------------------------------------------------
# match_atx_file
# ---------------------------------------------------------------------------

def match_atx_file(ts: GTime, prc_dir: str, popt: PrcOpt) -> str:
    """Locate ATX antenna phase-centre corrections file.

    Corresponds to ``match_atxfile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    dt   = datetime.datetime(int(ep[0]), int(ep[1]), int(ep[2]))

    gt_atx_1 = datetime.datetime(2006, 11, 5)
    gt_atx_2 = datetime.datetime(2011, 4, 17)
    gt_atx_3 = datetime.datetime(2017, 1, 29)

    ac = popt.ac_name.lower()
    if ac in ("gbm", "wum"):
        fname = ("igs08_" + ac + ".atx" if dt < gt_atx_3
                 else f"igs14_{popt.atx_week:04d}.atx")
    else:
        if dt < gt_atx_1:
            fname = "igs01_igs.atx"
        elif dt < gt_atx_2:
            fname = "igs05_igs.atx"
        elif dt < gt_atx_3:
            fname = "igs08_igs.atx"
        else:
            fname = f"igs14_{popt.atx_week:04d}.atx"

    candidate = os.path.join(prc_dir, "atx", fname)
    if os.path.isfile(candidate):
        return candidate
    print(f"Miss atx file: {candidate}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# match_base_obs_file
# ---------------------------------------------------------------------------

def match_base_obs_file(opt: PrcOpt, ts: GTime, prc_dir: str) -> str:
    """Locate base-station observation file.

    Corresponds to ``match_baseofile()`` in match_file.cc.
    """
    ep   = time2epoch(ts)
    yyyy = int(ep[0])
    doy  = int(time2doy(ts))
    obs_sub = opt.obsdir if opt.obsdir else "obs"
    obs_dir = os.path.join(opt.prcdir, f"{yyyy:04d}", f"{doy:03d}", obs_sub)

    if not os.path.isdir(obs_dir):
        return ""

    for fname in os.listdir(obs_dir):
        if "base" in fname:
            return os.path.join(obs_dir, fname)
    return ""


# ---------------------------------------------------------------------------
# load_prc_files  — top-level file loader
# ---------------------------------------------------------------------------

def load_prc_files(prc_dir: str, popt: PrcOpt, fopt: FilOpt) -> bool:
    """Load all ancillary product file paths into *fopt*.

    Corresponds to ``loadprcfiles()`` in match_file.cc.

    Returns
    -------
    bool — True on success, False if a mandatory file is missing.
    """
    ts = popt.ts
    te = popt.te

    no_gnss = (PMODE_INS_MECH <= popt.mode <= PMODE_LC_POS and
               popt.insopt.imu_align <= 0)

    # Navigation
    if not no_gnss:
        fopt.navf = match_nav_file(popt, ts, prc_dir)
        if not any(fopt.navf):
            print("Miss navigation file", file=sys.stderr)
            return False

    # Precise orbit + clock
    if not no_gnss and popt.sateph == EPHOPT_PREC:
        fopt.sp3f = match_sp3_file(popt, ts, prc_dir)
        if not fopt.sp3f[1]:
            return False
        fopt.clkf = match_clk_file(popt, ts, prc_dir)
        if not fopt.clkf[1]:
            return False

    # Code biases
    if not no_gnss:
        if popt.cbiaopt in (CBIAS_OPT_BRD_TGD, CBIAS_OPT_COD_DCB):
            fopt.dcb = match_dcb_file(popt, ts, prc_dir)
        elif popt.cbiaopt in (CBIAS_OPT_IGG_DCB, CBIAS_OPT_GBM_DCB):
            fopt.mgexdcb = match_mgexdcb_file(ts, prc_dir, popt.cbiaopt)
        elif popt.cbiaopt == CBIAS_OPT_MIX_DCB:
            fopt.dcb     = match_dcb_file(popt, ts, prc_dir)
            fopt.mgexdcb = match_mgexdcb_file(ts, prc_dir, popt.cbiaopt)

    # Ionosphere
    if not no_gnss and popt.ionoopt in (IONOOPT_TEC, IONOOPT_UC_CONS):
        fopt.iono = match_ion_file(ts, prc_dir)

    # PPP-specific
    ppp = (
        (PMODE_PPP_KINEMA <= popt.mode <= PMODE_PPP_FIXED or
         popt.mode in (PMODE_TC_PPP, PMODE_LC_PPP, PMODE_STC_PPP))
        and not no_gnss
    )
    if not no_gnss and ppp:
        atx = match_atx_file(ts, prc_dir, popt)
        if not atx:
            return False
        fopt.atx = atx

        if popt.modear in (ARMODE_PPPAR, ARMODE_PPPAR_ILS):
            if popt.arprod == AR_PROD_FCB:
                fopt.fcb = match_fcb_file(popt, ts, prc_dir)
                if not fopt.fcb:
                    return False
            elif AR_PROD_OSB_GRM <= popt.arprod <= AR_PROD_OSB_CNT:
                fopt.bia = match_bia_file(popt, ts, prc_dir)
                if not fopt.bia:
                    return False
            elif popt.arprod == AR_PROD_UPD:
                fopt.updf = match_upd_file(popt, ts, prc_dir)
                if not fopt.updf[2]:
                    return False

    # Tide corrections
    if not no_gnss and (popt.tidecorr & 2):
        fopt.blq = match_blq_file(prc_dir)
        fopt.eop = match_eop_file(ts, prc_dir, popt)
        if not fopt.eop:
            return False

    # PPK base observation
    ppk = (
        (PMODE_DGPS <= popt.mode <= PMODE_STATIC_START or
         popt.mode in (PMODE_TC_DGPS, PMODE_TC_PPK,
                       PMODE_LC_DGPS, PMODE_LC_PPK, PMODE_STC_PPK) or
         popt.insopt.imu_align in (INS_ALIGN_GNSS_PPK, INS_ALIGN_GNSS_DGPS))
        and not no_gnss
    )
    if not no_gnss and ppk:
        fopt.bobsf = match_base_obs_file(popt, ts, prc_dir)
        if not fopt.bobsf:
            print("Miss base obs file", file=sys.stderr)
            return False

    return True


# ---------------------------------------------------------------------------
# free_prc_files
# ---------------------------------------------------------------------------

def free_prc_files(fopt: FilOpt) -> None:
    """Reset all file paths in *fopt*.

    Corresponds to ``freeprcfiles()`` in match_file.cc.
    """
    fopt.navf    = ["", "", ""]
    fopt.sp3f    = ["", "", ""]
    fopt.clkf    = ["", "", ""]
    fopt.updf    = ["", "", ""]
    fopt.dcb     = ""
    fopt.mgexdcb = ""
    fopt.iono    = ""
    fopt.eop     = ""
    fopt.blq     = ""
    fopt.atx     = ""
    fopt.bia     = ""
    fopt.fcb     = ""
    fopt.bobsf   = ""


# ---------------------------------------------------------------------------
# matchout  — determine output file paths
# ---------------------------------------------------------------------------

def matchout(popt: PrcOpt, prc_dir: str, fopt: FilOpt, sopt: SolOpt) -> None:
    """Determine output solution file paths and write them into *fopt*.

    Corresponds to ``matchout()`` in match_file.cc.
    """
    sep   = os.sep
    ep    = time2epoch(popt.ts)
    yyyy  = int(ep[0])
    doy   = int(time2doy(popt.ts))
    mode_str = (KPMODESTR[popt.mode]
                if popt.mode < len(KPMODESTR) else f"MODE{popt.mode}")
    no_gnss = PMODE_INS_MECH <= popt.mode <= PMODE_LC_POS

    out_dir = os.path.join(prc_dir, f"{yyyy:04d}", f"{doy:03d}",
                           f"result_{mode_str}", "")

    if not no_gnss:
        sys_str = ""
        if popt.navsys & SYS_GPS: sys_str += "G"
        if popt.navsys & SYS_GLO: sys_str += "R"
        if popt.navsys & SYS_GAL: sys_str += "E"
        if popt.navsys & (SYS_CMP | SYS_BD3):
            if   popt.bd3opt == BD3OPT_OFF:   sys_str += "B2"
            elif popt.bd3opt == BD3OPT_BD23:   sys_str += "C"
            elif popt.bd3opt == BD3OPT_BD2_3:  sys_str += "B2B3"
            elif popt.bd3opt == BD3OPT_BD3:    sys_str += "B3"
        if popt.navsys & SYS_QZS: sys_str += "J"

        out_name = f"{popt.site_name}_{sys_str}"

        # Frequency suffix
        freq_map = {1: "SF", 2: "DF", 3: "TF", 4: "QF"}
        out_name += "_" + freq_map.get(popt.nf, "")

        # Ionosphere suffix
        from constants import (IONOOPT_IFLC, IONOOPT_IF2, IONOOPT_UC,
                                IONOOPT_UC_CONS)
        iono_map = {IONOOPT_IFLC: "IF", IONOOPT_IF2: "IF2",
                    IONOOPT_UC: "UC", IONOOPT_UC_CONS: "UC_CON"}
        if popt.ionoopt in iono_map:
            out_name += "_" + iono_map[popt.ionoopt]

        # AR suffix
        ar = popt.modear > ARMODE_OFF
        if ar:
            if popt.modear == ARMODE_PPPAR_ILS:
                prod_tag = {
                    AR_PROD_IRC: "_FIX_IRC", AR_PROD_FCB: "_FIX_FCB",
                    AR_PROD_UPD: "_FIX_UPD",
                }.get(popt.arprod, "_FIX_OSB")
                out_name += prod_tag
            else:
                ar_tag = {
                    ARMODE_CONT:    "_FIX",
                    ARMODE_INST:    "_INST",
                    ARMODE_FIXHOLD: "_HOLD",
                }.get(popt.modear, "_FIX")
                out_name += ar_tag
        else:
            out_name += "_FLOAT"

        full_out_dir = os.path.join(out_dir, popt.obsdir, "")
        os.makedirs(full_out_dir, exist_ok=True)

        fopt.solf = os.path.join(full_out_dir, popt.ac_name,
                                 f"{out_name}.pos")

        if popt.insopt.rts:
            fopt.rts_ins_fw = "rts.bin"
            fopt.rtsfile    = os.path.join(full_out_dir, popt.ac_name,
                                           f"{out_name}.rts")

        if (sopt.ambres and
                PMODE_PPP_KINEMA <= popt.mode <= PMODE_PPP_FIXED and
                popt.modear in (ARMODE_PPPAR, ARMODE_PPPAR_ILS)):
            fopt.wl_amb = os.path.join(full_out_dir, popt.ac_name,
                                       f"{out_name}.wlamb")
            fopt.nl_amb = os.path.join(full_out_dir, popt.ac_name,
                                       f"{out_name}.nlamb")
            fopt.lc_amb = os.path.join(full_out_dir, popt.ac_name,
                                       f"{out_name}.lcamb")
    else:
        full_out_dir = os.path.join(out_dir, popt.obsdir, "")
        os.makedirs(full_out_dir, exist_ok=True)
        from constants import PMODE_INS_MECH, PMODE_LC_POS
        if popt.mode == PMODE_INS_MECH:
            fopt.solf = os.path.join(full_out_dir, "INS_MECH.pos")
        elif popt.mode == PMODE_LC_POS:
            fopt.solf = os.path.join(full_out_dir, "LC_POS.pos")
        if popt.insopt.rts:
            fopt.rts_ins_fw = "rts.bin"
            fopt.rtsfile    = os.path.join(full_out_dir, "LC_SOL.rts")


# ---------------------------------------------------------------------------
# parse_cmd  — command-line argument parser
# ---------------------------------------------------------------------------

def parse_cmd(argv: list, popt: PrcOpt, sopt: SolOpt, fopt: FilOpt
              ) -> Tuple[bool, int]:
    """Parse the PPP-AR command-line arguments.

    Corresponds to ``parsecmd()`` + ``loadconf()`` in match_file.cc.

    Parameters
    ----------
    argv  : list[str] — command-line arguments (without the program name).
    popt  : PrcOpt    — modified in-place.
    sopt  : SolOpt    — modified in-place.
    fopt  : FilOpt    — modified in-place.

    Returns
    -------
    (ok, port) — ok is True on success; port is the optional server port.
    """
    from constants import (
        SYS_GPS, SYS_GLO, SYS_GAL, SYS_CMP, SYS_BD3, SYS_QZS,
        BD3OPT_OFF, BD3OPT_BD2_3, BD3OPT_BD23, BD3OPT_BD3,
        IONOOPT_UC, IONOOPT_UC_CONS,
        KPMODESTR, ARMODE_OFF,
    )

    mode_str  = ""
    conf_file = ""
    mask      = SYS_NONE
    level     = 128
    ar        = ARMODE_OFF
    port      = 0
    i = 0

    while i < len(argv):
        arg = argv[i]
        if arg == "-C" and i + 1 < len(argv):
            i += 1
            conf_file = argv[i]
            if not conf_file or not os.path.isfile(conf_file):
                print(f"OPEN CONFIGURATION FILE ERROR file={conf_file}",
                      file=sys.stderr)
                return False, port
            print(f"Config file: {conf_file} (option parsing not implemented — "
                  "extend load_conf() to read your .conf format)",
                  file=sys.stderr)
        elif arg == "-M" and i + 1 < len(argv):
            i += 1
            mode_str = argv[i]
        elif arg == "-P" and i + 1 < len(argv):
            i += 1
            port = int(argv[i])
        elif arg == "-S" and i + 1 < len(argv):
            i += 1
            for c in argv[i]:
                if   c == 'G': mask |= SYS_GPS
                elif c == 'R': mask |= SYS_GLO
                elif c == 'E': mask |= SYS_GAL
                elif c == 'C':
                    if popt.bd3opt == BD3OPT_OFF:
                        mask |= SYS_CMP
                    elif popt.bd3opt in (BD3OPT_BD2_3, BD3OPT_BD23):
                        mask |= SYS_CMP | SYS_BD3
                    elif popt.bd3opt == BD3OPT_BD3:
                        mask |= SYS_BD3
                elif c == 'J': mask |= SYS_QZS
            if mask == SYS_NONE:
                print(f"SATELLITE SYSTEM SET ERROR: {argv[i]}", file=sys.stderr)
                return False, port
        elif arg == "-L" and i + 1 < len(argv):
            i += 1
            level = int(argv[i])
        elif arg == "-A" and i + 1 < len(argv):
            i += 1
            ar = int(argv[i])
        i += 1

    # Single-frequency PPP should use GIM
    if popt.nf == 1 and popt.ionoopt == IONOOPT_UC:
        popt.ionoopt = IONOOPT_UC_CONS

    if mask:
        popt.navsys = mask

    if mode_str:
        if mode_str in KPMODESTR:
            popt.mode = KPMODESTR.index(mode_str)
        else:
            print(f"Unknown mode: {mode_str}", file=sys.stderr)
            return False, port

    sopt.trace   = level
    popt.modear  = ar
    return True, port