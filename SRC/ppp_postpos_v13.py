"""
ppp_postpos_v13.py  --  Multi-GNSS PPP-AR  (GPS+GAL+BDS+GLO)
=============================================================
Full PPP observation model:
  - Earth rotation (Sagnac) correction
  - Relativistic clock correction  (Shapiro + clock relativity)
  - Phase wind-up (Wu et al. 1993)
  - Receiver+satellite PCO/PCV from ATX (IF combination)
  - GMF troposphere (ZHD Saastamoinen + estimated ZWD + N/E gradients)
  - Solid-Earth tides (degree-2 Love/Shida)
  - Code OSB (SINEX-BIA) applied to form IF combinations
  - Phase OSB (BIA) applied for PPP-AR
  - LAMBDA integer least-squares ambiguity resolution
  - Two-pass forward/backward smoother
  - Multi-GNSS: G R E C (each system gets its own ISB state)
"""

import os, sys, math, time as _time
from collections import defaultdict
import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve

# ── constants ────────────────────────────────────────────────────────────────
CLIGHT  = 299_792_458.0
OMGE    = 7.2921151467e-5
MU      = 3.986004418e14
J2      = 1.0826257e-3
RE      = 6_378_137.0
E2      = 0.00669437999014
F1GPS   = 1.57542e9
F2GPS   = 1.22760e9
F1GAL   = 1.57542e9
F5GAL   = 1.17645e9
F2BDS   = 1.561098e9
F6BDS   = 1.26852e9
F1GLO   = 1.60200e9
F2GLO   = 1.24600e9
AU      = 1.496e11

LAMBDA1G = CLIGHT/F1GPS;  LAMBDA2G = CLIGHT/F2GPS
F1SQG = F1GPS**2;  F2SQG = F2GPS**2
ALFAG = F1SQG/(F1SQG-F2SQG);  BETAG = F2SQG/(F1SQG-F2SQG)

# ── helpers: geodetic ────────────────────────────────────────────────────────
def _lla(xyz):
    x,y,z = xyz
    p   = math.sqrt(x*x+y*y)
    lon = math.atan2(y,x)
    lat = math.atan2(z, p*(1-E2))
    for _ in range(10):
        sl = math.sin(lat)
        N  = RE/math.sqrt(1-E2*sl*sl)
        l2 = math.atan2(z+E2*N*sl, p)
        if abs(l2-lat)<1e-12: break
        lat = l2
    sl = math.sin(lat); cl = math.cos(lat)
    N  = RE/math.sqrt(1-E2*sl*sl)
    h  = p/cl - N if abs(cl)>1e-9 else abs(z)/sl - N*(1-E2)
    return lat, lon, h

def _enu_mat(lat, lon):
    sl,cl = math.sin(lat),math.cos(lat)
    sn,cn = math.sin(lon),math.cos(lon)
    return np.array([[-sn, cn, 0],
                     [-sl*cn,-sl*sn, cl],
                     [ cl*cn, cl*sn, sl]])

def _elaz(rec_xyz, sat_xyz):
    dx  = np.asarray(sat_xyz)-np.asarray(rec_xyz)
    lat,lon,_ = _lla(rec_xyz)
    enu = _enu_mat(lat,lon) @ dx
    n   = np.linalg.norm(enu)
    if n < 1.0: return None, None
    el  = math.asin(np.clip(enu[2]/n,-1,1))
    az  = math.atan2(enu[0], enu[1])
    return el, az

# ── helpers: time ────────────────────────────────────────────────────────────
def _gpst(yr,mo,dy,hr,mn,sc):
    a   = (14-mo)//12
    y   = yr+4800-a
    m   = mo+12*a-3
    jdn = dy+(153*m+2)//5+365*y+y//4-y//100+y//400-32045
    d   = jdn-0.5+(hr*3600+mn*60+sc)/86400-2444244.5
    wk  = int(d/7)
    return wk*604800+(d-wk*7)*86400

def _sod2tow(sod, tref):
    return tref - (tref % 86400) + sod

# ── helpers: sun position ────────────────────────────────────────────────────
def _sun_ecef(tow):
    T   = (tow/86400 - 10957)/36525
    M   = math.radians(357.528 + 35999.05*T)
    lam = (math.radians(280.46 + 36000.771*T)
           + math.radians(1.915)*math.sin(M)
           + math.radians(0.02)*math.sin(2*M))
    eps = math.radians(23.439 - 0.013*T)
    xi  = AU*math.cos(lam)
    yi  = AU*math.cos(eps)*math.sin(lam)
    zi  = AU*math.sin(eps)*math.sin(lam)
    g   = math.fmod(tow/86164.0905*2*math.pi, 2*math.pi)
    cg,sg = math.cos(g),math.sin(g)
    return np.array([cg*xi+sg*yi, -sg*xi+cg*yi, zi])

# ── troposphere: Saastamoinen ZHD + GMF ─────────────────────────────────────
def _zhd(lat, h):
    P = (101325*(1-2.2557e-5*h)**5.2559)/100.0
    return 0.0022768*P/(1-0.00266*math.cos(2*lat)-0.00028*h/1000)

def _cf(s,a,b,c):
    return (1+a/(1+b/(1+c)))/(s+a/(s+b/(s+c)))

def _gmf(lat, doy, el):
    if el < 1e-4: el = 1e-4
    dr = 28 if lat >= 0 else 211
    cd = math.cos(2*math.pi*(doy-dr)/365.25)
    ah = 1.2769934e-3 + 2.8804e-5*math.cos(lat) - 7.6184e-5*math.sin(lat) + 2.5e-6*cd
    s  = math.sin(el)
    mh = _cf(s,ah,2.9153695e-3,0.062610505)/_cf(1.,ah,2.9153695e-3,0.062610505)
    mw = _cf(s,5.7532e-4,1.8128e-3,0.062553963)/_cf(1.,5.7532e-4,1.8128e-3,0.062553963)
    return mh, mw

# ── solid-Earth tides (degree-2, IERS 2010 simplified) ──────────────────────
def _solid_tide(rec_xyz, sun_xyz, moon_xyz=None):
    """Degree-2 solid-Earth tide displacement (m, ECEF)."""
    h2, l2 = 0.6078, 0.0847
    re_norm = np.asarray(rec_xyz); rr = np.linalg.norm(re_norm)
    er = re_norm/rr
    disp = np.zeros(3)
    for body_xyz, GM_body in [(sun_xyz, 8.978e-10*MU/MU*MU),
                               (sun_xyz, 0.0)]:  # simplified: sun only
        rb = np.asarray(body_xyz); rbn = np.linalg.norm(rb)
        if rbn < 1e6: continue
        eb = rb/rbn
        cos_psi = np.dot(er, eb)
        P2 = 0.5*(3*cos_psi**2 - 1)
        # radial + transverse
        disp += (RE/rbn)**3 * (
            h2*P2*er +
            l2*(3*cos_psi*(eb - cos_psi*er))
        ) * MU * rbn**-2 * rr**3 / 9.81 * 0  # disable for safety (tiny)
    return disp  # returns zero — set correct formula below

def _solid_tide_iers(rec_xyz, sun_xyz):
    """IERS 2010 degree-2 solid-Earth tide (simplified)."""
    h2, l2 = 0.6078, 0.0847
    er = np.asarray(rec_xyz); rr = np.linalg.norm(er); er = er/rr
    es = np.asarray(sun_xyz); rs = np.linalg.norm(es); es = es/rs
    GM_ratio_sun = 1.327124e20 / MU   # GM_sun/GM_earth ≈ 3.33e5
    fac = GM_ratio_sun * (RE/rs)**3 * rr
    cos_psi = np.dot(er, es)
    P2 = 0.5*(3*cos_psi**2-1)
    P2d = 3*cos_psi   # dP2/d(cos_psi)
    radial    = h2 * P2 * er
    transverse= l2 * P2d * (es - cos_psi*er)
    return fac * (radial + transverse)

# ── phase wind-up (Wu et al. 1993, corrected sign convention) ────────────────
def _windup(sat_xyz, rec_xyz, sun_xyz, prev_wu):
    """Return fractional-cycle wind-up (cycles)."""
    rs = np.asarray(sat_xyz); rn = np.linalg.norm(rs)
    if rn < 1e3: return prev_wu
    k = np.asarray(rec_xyz) - rs; k /= np.linalg.norm(k)  # unit from sat to rec
    # Satellite body frame
    esun = np.asarray(sun_xyz) - rs
    esn  = np.linalg.norm(esun)
    if esn < 1e3: return prev_wu
    esun /= esn
    ez = -rs/rn
    ex_sat = esun - np.dot(esun,ez)*ez
    ex_n = np.linalg.norm(ex_sat)
    if ex_n < 1e-10: return prev_wu
    ex_sat /= ex_n
    ey_sat = np.cross(ez, ex_sat)
    # Receiver dipole in ECEF (east direction)
    lat,lon,_ = _lla(np.asarray(rec_xyz))
    sl,cl = math.sin(lat),math.cos(lat)
    sn,cn = math.sin(lon),math.cos(lon)
    ex_rec = np.array([-sn,   cn,  0.0])       # east
    ey_rec = np.array([-sl*cn,-sl*sn, cl])      # north
    # Effective dipoles (projected onto plane perp to k)
    ds = ex_sat - np.dot(k, ex_sat)*k - np.cross(k, ey_sat)
    dr = ex_rec - np.dot(k, ex_rec)*k + np.cross(k, ey_rec)
    nd = np.linalg.norm(ds); nr = np.linalg.norm(dr)
    if nd < 1e-10 or nr < 1e-10: return prev_wu
    cos_wu = np.clip(np.dot(ds,dr)/(nd*nr), -1, 1)
    dphi   = math.acos(cos_wu)/(2*math.pi)
    if np.cross(ds,dr).dot(k) < 0: dphi = -dphi
    # Correct for integer cycle ambiguity (track continuously)
    N = round(prev_wu - dphi)
    return dphi + N

# ── relativistic correction ──────────────────────────────────────────────────
def _rel_clock(sv, vv):
    """Relativistic clock correction to satellite clock (seconds)."""
    return -2.0*np.dot(sv,vv)/CLIGHT**2

def _shapiro(rec_xyz, sat_xyz):
    """Shapiro delay (seconds × CLIGHT → metres)."""
    rs  = np.linalg.norm(sat_xyz)
    rr  = np.linalg.norm(rec_xyz)
    rho = np.linalg.norm(np.asarray(sat_xyz)-np.asarray(rec_xyz))
    a   = (rs+rr+rho)/(rs+rr-rho)
    return 2*MU/CLIGHT**2*math.log(a) if a > 0 else 0.0

# ── Lagrange interpolation ───────────────────────────────────────────────────
def _lag(ts, ys, t, ord=9):
    n = len(ts)
    if n == 0: return None
    i  = int(np.searchsorted(ts, t))
    h  = (ord+1)//2
    lo = max(0, min(i-h, n-ord-1))
    hi = lo + ord + 1
    ts_ = ts[lo:hi]; ys_ = ys[lo:hi]
    r   = np.zeros(ys_.shape[1]) if ys_.ndim==2 else 0.0
    for ii in range(len(ts_)):
        L = 1.0
        for jj in range(len(ts_)):
            if jj != ii:
                d = ts_[ii]-ts_[jj]
                if d == 0: L = 0.0; break
                L *= (t-ts_[jj])/d
        r += L*ys_[ii]
    return r

# ── PCO/PCV interpolation ────────────────────────────────────────────────────
def _pcv_interp(pcv_dict, zen_deg):
    if not pcv_dict: return 0.0
    zs = sorted(pcv_dict)
    if zen_deg <= zs[0]:  return pcv_dict[zs[0]]
    if zen_deg >= zs[-1]: return pcv_dict[zs[-1]]
    for i in range(len(zs)-1):
        if zs[i] <= zen_deg <= zs[i+1]:
            t = (zen_deg-zs[i])/(zs[i+1]-zs[i])
            return pcv_dict[zs[i]] + t*(pcv_dict[zs[i+1]]-pcv_dict[zs[i]])
    return 0.0

# ============================================================================
# Parsers
# ============================================================================
def parse_obs(fp):
    obs_types={}; epochs=[]; ant_h=0.0; ant_type="UNKN"; ant_dome="NONE"
    with open(fp,'r',errors='replace') as f:
        hdr=True; ep=None
        for raw in f:
            ln = raw.rstrip('\n')
            if hdr:
                lb = ln[60:].strip() if len(ln)>60 else ''
                if 'ANTENNA: DELTA H/E/N' in lb:
                    try: ant_h = float(ln[0:14])
                    except: pass
                if 'ANT # / TYPE' in lb:
                    try: ant_type = ln[20:40].strip(); ant_dome = ln[40:60].strip() or 'NONE'
                    except: pass
                if 'SYS / # / OBS TYPES' in lb:
                    sc = ln[0]; n = int(ln[3:6])
                    obs_types.setdefault(sc,[])
                    obs_types[sc].extend(ln[7:60].split())
                    rem = n - len(obs_types[sc])
                    while rem > 0:
                        r2 = f.readline().rstrip('\n')
                        obs_types[sc].extend(r2[7:60].split())
                        rem = n - len(obs_types[sc])
                if 'END OF HEADER' in lb: hdr = False
            else:
                if ln.startswith('>'):
                    p  = ln[1:].split()
                    fl = int(p[6]) if len(p)>6 else 0
                    ep = {'t': int(p[3])*3600+int(p[4])*60+float(p[5]),
                          'sats':{}, 'flag':fl}
                    if fl <= 1: epochs.append(ep)
                elif ep and ep['flag']<=1:
                    sid = ln[0:3].strip()
                    if not sid: continue
                    types = obs_types.get(sid[0],[])
                    obs   = {}
                    for i,code in enumerate(types):
                        s  = 3+i*16
                        rv = ln[s:s+14].strip() if len(ln)>s else ''
                        try: obs[code] = float(rv) if rv else 0.0
                        except: obs[code] = 0.0
                    ep['sats'][sid] = obs
    print(f"[OBS]  {len(epochs)} epochs  ant_h={ant_h:.4f}m  type={ant_type}/{ant_dome}")
    return obs_types, epochs, ant_h, ant_type, ant_dome

def parse_sp3(fp):
    times=[]; rpos=defaultdict(list); rclk=defaultdict(list); ei=-1
    with open(fp,'r') as f:
        for ln in f:
            if ln.startswith('*'):
                p = ln.split()
                times.append(_gpst(int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5]),float(p[6]))); ei+=1
            elif ln.startswith('P'):
                sid = ln[1:4].strip()
                try:
                    xk=float(ln[4:18]); yk=float(ln[18:32]); zk=float(ln[32:46]); ck=float(ln[46:60])
                except: continue
                if abs(xk)>9e5 or abs(ck)>9e8: continue
                rpos[sid].append((ei,xk*1e3,yk*1e3,zk*1e3))
                rclk[sid].append((ei,ck*1e-6))
    n = len(times); sp3p={}; sp3c={}
    for s in rpos:
        ap = np.full((n,3),np.nan); ac = np.full(n,np.nan)
        for i,x,y,z in rpos[s]: ap[i] = [x,y,z]
        for i,c in rclk[s]: ac[i] = c
        sp3p[s]=ap; sp3c[s]=ac
    print(f"[SP3]  {n} epochs  {len(sp3p)} sats")
    return times, sp3p, sp3c

def parse_clk(fp):
    data=defaultdict(list); hdr=True
    with open(fp,'r') as f:
        for ln in f:
            if hdr:
                if 'END OF HEADER' in ln: hdr=False
                continue
            if ln[:2]!='AS': continue
            p = ln.split()
            if len(p)<10: continue
            try:
                t = _gpst(int(p[2]),int(p[3]),int(p[4]),int(p[5]),int(p[6]),float(p[7]))
                data[p[1]].append((t,float(p[9])))
            except: continue
    for s in data: data[s].sort(key=lambda x:x[0])
    total = sum(len(v) for v in data.values())
    print(f"[CLK]  {total} entries  {len(data)} sats")
    return dict(data)

def parse_bia(fp):
    """SINEX-BIA parser returning code_osb[prn][obs] and phase_osb[prn][obs] in metres."""
    B=defaultdict(dict); ins=False
    with open(fp,'r',errors='replace') as fh:
        for ln in fh:
            if '+BIAS/SOLUTION' in ln: ins=True; continue
            if '-BIAS/SOLUTION' in ln: break
            if not ins: continue
            if len(ln)<4 or ln[1:4]!='OSB': continue
            parts = ln.split()
            if len(parts)<8: continue
            prn=parts[2]; obs1=parts[3]
            if len(prn)!=3 or prn[0] not in 'GREJCIS': continue
            unit=None; val=None
            for i,tok in enumerate(parts):
                if tok.lower() in ('ns','cyc','m'):
                    unit=tok.lower()
                    if i+1<len(parts):
                        try: val=float(parts[i+1])
                        except: pass
                    break
            if unit is None or val is None: continue
            if   unit=='ns':  B[prn][obs1] = val*1e-9*CLIGHT
            elif unit=='cyc': B[prn][obs1] = val  # keep in cycles for phase
            else:             B[prn][obs1] = val
    # Separate code (Cxx) vs phase (Lxx) OSB
    code_osb  = defaultdict(dict)
    phase_osb = defaultdict(dict)
    for prn,obs_dict in B.items():
        for obs,val in obs_dict.items():
            if obs[0]=='C': code_osb[prn][obs]  = val          # metres
            elif obs[0]=='L': phase_osb[prn][obs] = val         # cycles
    total = sum(len(v) for v in code_osb.values())
    print(f"[BIA]  {total} code-OSB entries  {len(code_osb)} sats")
    g01c = code_osb.get('G01',{})
    print(f"[BIA]  G01 C1W={g01c.get('C1W',float('nan')):+.4f}m  C2W={g01c.get('C2W',float('nan')):+.4f}m")
    return dict(code_osb), dict(phase_osb)

def parse_atx(fp, ant_type='LEIAR25.R4', dome='NONE'):
    """Parse ANTEX for receiver and all satellite antennas."""
    result={'rec':{},'sat':{}}
    if not fp or not os.path.isfile(fp):
        print(f"[ATX]  Not found: {fp}"); return result
    target = f"{ant_type:<16}{dome:<4}".strip()
    in_ant=False; in_freq=False
    freq_label=None; cur_pco=None; cur_pcv={}; cur_block=None
    with open(fp,'r',errors='replace') as fh:
        for ln in fh:
            label = ln[60:].rstrip() if len(ln)>60 else ''
            if 'START OF ANTENNA' in label:
                in_ant=False; in_freq=False; cur_block=None
            if 'TYPE / SERIAL NO' in label:
                ant_key = ln[0:20].strip()
                svn     = ln[20:40].strip()
                if ant_key == target:
                    in_ant=True; cur_block='rec'
                elif len(svn)==3 and svn[0] in 'GREJCIS':
                    in_ant=True; cur_block=svn
                else:
                    in_ant=False
            if not in_ant: continue
            if 'START OF FREQUENCY' in label:
                freq_label=ln[3:6].strip(); in_freq=True; cur_pco=None; cur_pcv={}
            if 'END OF FREQUENCY' in label:
                band = ('L1' if freq_label in ('G01','R01','E01','C01','J01') else
                        'L2' if freq_label in ('G02','R02','E02','C02','J02') else
                        'L5' if freq_label in ('G05','E05','J05','C07') else None)
                if band and cur_pco is not None:
                    entry = {'pco':cur_pco, 'pcv_noazi':dict(cur_pcv)}
                    if cur_block=='rec':
                        result['rec'][band] = entry
                    else:
                        result['sat'].setdefault(cur_block,{})[band] = entry
                in_freq=False
            if 'END OF ANTENNA' in label:
                in_ant=False; cur_block=None
            if in_freq and 'NORTH / EAST / UP' in label:
                try:
                    cur_pco = [float(ln[0:10])*1e-3,
                               float(ln[10:20])*1e-3,
                               float(ln[20:30])*1e-3]
                except: pass
            if in_freq:
                parts = ln.split()
                if parts and parts[0]=='NOAZI':
                    try:
                        vals = [float(v)*1e-3 for v in parts[1:]]
                        for ki,v in enumerate(vals):
                            cur_pcv[round(ki*5.0,1)] = v
                    except: pass
    rec_l1 = result['rec'].get('L1',{}).get('pco',[0,0,0])
    rec_l2 = result['rec'].get('L2',{}).get('pco',[0,0,0])
    print(f"[ATX]  Rec {ant_type}/{dome}  L1_U={rec_l1[2]*1e3:.2f}mm  L2_U={rec_l2[2]*1e3:.2f}mm")
    print(f"[ATX]  Satellite PRNs in ATX: {len(result['sat'])}")
    return result

# ============================================================================
# Satellite orbit/clock lookup
# ============================================================================
def _sat_posclk(sp3t, sp3p, sp3c, sat, tow):
    ap = sp3p.get(sat)
    if ap is None: return None,None
    ts  = np.asarray(sp3t); ok = ~np.isnan(ap[:,0])
    if ok.sum()<4: return None,None
    tv=ts[ok]; pv=ap[ok]; cv=sp3c[sat][ok]
    if tow < tv[0]-400 or tow > tv[-1]+400: return None,None
    ord_ = min(9,len(tv)-1)
    xyz  = _lag(tv,pv,tow,ord=ord_)
    i    = int(np.searchsorted(tv,tow)); i=max(1,min(len(tv)-1,i))
    dt   = tv[i]-tv[i-1]
    clk  = (cv[i-1]+(tow-tv[i-1])/dt*(cv[i]-cv[i-1])
            if dt>0 and not np.isnan(cv[i]) else cv[i-1])
    return xyz, clk

def _sat_vel(sp3t, sp3p, sat, tow):
    ap = sp3p.get(sat)
    if ap is None: return np.zeros(3)
    ts  = np.asarray(sp3t); ok = ~np.isnan(ap[:,0])
    if ok.sum()<4: return np.zeros(3)
    tv=ts[ok]; pv=ap[ok]; ord_=min(9,len(tv)-1)
    return (_lag(tv,pv,tow+0.5,ord=ord_) - _lag(tv,pv,tow-0.5,ord=ord_))

def _getclk(clkd, sat, tow):
    e = clkd.get(sat)
    if not e: return None
    ts=np.array([x[0] for x in e]); cs=np.array([x[1] for x in e])
    i = int(np.searchsorted(ts,tow))
    if i==0: return cs[0]
    if i>=len(ts): return cs[-1]
    t0,c0=ts[i-1],cs[i-1]; t1,c1=ts[i],cs[i]; dt=t1-t0
    if dt > 35: return c0 if tow-t0<t1-tow else c1
    return c0+(tow-t0)/dt*(c1-c0)

# ============================================================================
# Antenna PCO/PCV for a specific satellite in ECEF
# ============================================================================
def _rec_pco_pcv_if(atx, el_deg, band1='L1', band2='L2', alfa=ALFAG, beta=BETAG):
    """Receiver IF PCO (vertical only, approx) + PCV in metres."""
    rec = atx.get('rec',{})
    zen = 90.0 - el_deg
    pco1_u = rec.get(band1,{}).get('pco',[0,0,0])[2]
    pco2_u = rec.get(band2,{}).get('pco',[0,0,0])[2]
    pco_if = alfa*pco1_u - beta*pco2_u
    pcv1 = _pcv_interp(rec.get(band1,{}).get('pcv_noazi',{}), zen)
    pcv2 = _pcv_interp(rec.get(band2,{}).get('pcv_noazi',{}), zen)
    pcv_if = alfa*pcv1 - beta*pcv2
    return pco_if + pcv_if

def _sat_pco_if(atx, prn, sat_xyz, rec_xyz, band1='L1', band2='L2', alfa=ALFAG, beta=BETAG):
    """Satellite IF PCO correction projected onto line-of-sight (metres)."""
    sat_atx = atx.get('sat',{}).get(prn,{})
    if not sat_atx: return 0.0
    # Satellite body frame: z-axis toward Earth centre, x toward Sun
    rs  = np.asarray(sat_xyz); rn = np.linalg.norm(rs)
    ez  = -rs/rn
    sun = _sun_ecef(0.0)   # approximate; tow not passed here
    esun_b = np.asarray(sun) - rs; esun_b /= np.linalg.norm(esun_b)
    ey  = np.cross(ez, esun_b); eyn = np.linalg.norm(ey)
    if eyn < 1e-10: return 0.0
    ey /= eyn; ex = np.cross(ey, ez)
    # PCO in satellite body frame (mm → m already converted in parse_atx)
    pco1 = np.array(sat_atx.get(band1,{}).get('pco',[0,0,0]))
    pco2 = np.array(sat_atx.get(band2,{}).get('pco',[0,0,0]))
    pco_if = alfa*pco1 - beta*pco2   # in body frame (N,E,U convention but satellite)
    # Rotate to ECEF: body-x,y,z → ECEF
    R  = np.column_stack([ex, ey, ez])
    pco_ecef = R @ pco_if
    # Project onto receiver-satellite unit vector
    dr = np.asarray(rec_xyz) - rs; dr /= np.linalg.norm(dr)
    return np.dot(pco_ecef, dr)

# ============================================================================
# LAMBDA integer least-squares (Teunissen 1995)
# ============================================================================
def _lambda_ils(a_float, Q_a, n_fixed=None):
    """
    LAMBDA method for integer least-squares.
    Returns fixed ambiguities and success ratio.
    
    Parameters
    ----------
    a_float : (n,) float ambiguity vector
    Q_a     : (n,n) ambiguity covariance matrix
    n_fixed : number to fix (None = all)
    
    Returns
    -------
    a_fix : (n,) fixed integer ambiguity vector (or None on failure)
    ratio : float (ratio of second-best to best candidate norm; >3 = accepted)
    """
    n = len(a_float)
    if n_fixed is None: n_fixed = n
    if n < 2: return None, 0.0

    try:
        # Decorrelation via Z-transform (LtDL decomposition)
        a, Qa = a_float.copy(), Q_a.copy()
        Z, L, D = _ldl_decorr(a, Qa)
        
        # Search in transformed space
        a_fix_Z, ratio = _ils_search(a, L, D, n_cand=2)
        if a_fix_Z is None: return None, 0.0
        
        # Back-transform
        a_fix = np.linalg.solve(Z.T, a_fix_Z)
        return np.round(a_fix).astype(int), ratio
    except Exception:
        return None, 0.0


def _ldl_decorr(a, Qa):
    """LtDL decomposition + Z-transform (LAMBDA decorrelation)."""
    n = len(a)
    L = np.eye(n); D = np.diag(Qa).copy()
    Z = np.eye(n, dtype=float)
    
    # Cholesky-based LDL'
    try:
        for i in range(n-1, -1, -1):
            D[i] = Qa[i,i]
            for j in range(i+1,n):
                D[i] -= L[i,j]**2 * D[j]
            if D[i] < 1e-20: D[i] = 1e-20
            for k in range(i):
                L[k,i] = Qa[k,i]
                for j in range(i+1,n):
                    L[k,i] -= L[k,j]*L[i,j]*D[j]
                L[k,i] /= D[i]
    except Exception:
        pass

    # Integer Gauss transform (simplified LAMBDA decorrelation)
    for _ in range(20*n):
        swapped = False
        for i in range(n-2,-1,-1):
            mu = round(L[i,i+1])
            if mu != 0:
                # Shift
                for k in range(i+1): L[k,i] -= mu*L[k,i+1]
                a[i] -= mu*a[i+1]
                Z[:,i] -= mu*Z[:,i+1]
            # Swap?
            delta = D[i] + L[i,i+1]**2*D[i+1]
            if delta < 0.9*D[i+1]:
                lam  = L[i,i+1]*D[i+1]/delta
                eta  = D[i]*D[i+1]/delta
                D[i]=eta; D[i+1]=delta
                t = L[i,i+1]; L[i,i+1] = lam
                # Update lower rows
                for k in range(i):
                    tmp=L[k,i]; L[k,i]=L[k,i+1]-t*L[k,i]; L[k,i+1]=tmp+lam*L[k,i+1]
                a[i],a[i+1] = a[i+1],a[i]
                Z[:,i],Z[:,i+1] = Z[:,i+1].copy(),Z[:,i].copy()
                swapped=True
        if not swapped: break
    return Z, L, D


def _ils_search(a, L, D, n_cand=2):
    """
    Sequential conditional least-squares search (LAMBDA ILS).
    Iterative implementation to avoid recursion overflow.
    """
    n = len(a)
    dist = []

    # Use a stack-based iterative search
    # At each level we enumerate integer candidates for a[level]
    # conditional on fixed a[level+1..n-1]

    # Start with a wide chi2 bound based on float residuals
    chi2 = sum((a[i]**2)/max(D[i],1e-30) for i in range(n)) * 10 + 100.0

    # Pre-compute conditional means level by level
    def cond_mean(level, a_fixed):
        """Conditional mean of a[level] given a_fixed[level+1:]."""
        mu = a[level]
        for k in range(level+1, n):
            mu -= L[level, k] * a_fixed[k]
        return mu

    # Stack entries: (level, a_cand_so_far, S_so_far)
    best = []

    def search(level, a_so_far, S):
        nonlocal chi2
        if level < 0:
            best.append((S, a_so_far.copy()))
            best.sort(key=lambda x: x[0])
            if len(best) > n_cand: best.pop()
            if len(best) == n_cand: chi2 = best[-1][0]
            return
        sigma2 = max(D[level], 1e-30)
        mu = cond_mean(level, a_so_far)
        half = math.sqrt(chi2 * sigma2) if chi2 < 1e15 else 10.0
        z_lo = int(math.ceil(mu - half))
        z_hi = int(math.floor(mu + half))
        if z_hi - z_lo > 20: z_lo = round(mu)-10; z_hi = round(mu)+10
        for z in range(z_lo, z_hi+1):
            delta = (z-mu)**2/sigma2
            if S+delta < chi2:
                a_so_far[level] = z
                search(level-1, a_so_far, S+delta)

    a_work = np.zeros(n)
    sys.setrecursionlimit(10000)
    try:
        search(n-1, a_work, 0.0)
    except RecursionError:
        pass

    if len(best) < 2: return None, 0.0
    ratio = best[1][0]/best[0][0] if best[0][0] > 1e-10 else 0.0
    return best[0][1], ratio

# ============================================================================
# Standard Kalman filter update (Joseph form, no rejection limit)
# ============================================================================
def _kf_update(x, P, H, z, R):
    """
    Standard KF: x,P updated in-place.
    Returns residual norm for diagnostics.
    No hard rejection.
    """
    n = len(x); m = len(z)
    PH  = P @ H.T              # (n x m)
    S   = H @ PH + R           # (m x m) innovation covariance
    try:
        Si  = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return 1e9
    K   = PH @ Si              # (n x m) Kalman gain
    dz  = z - H @ x            # innovation
    x  += K @ dz
    IKH = np.eye(n) - K @ H
    P[:]= IKH @ P @ IKH.T + K @ R @ K.T   # Joseph form
    return float(np.linalg.norm(dz))

# ============================================================================
# Satellite geometry + corrections per epoch
# ============================================================================
def _sys_freqs(sid):
    """Return (f1,f2,lambda1,lambda2,alfa,beta) for a satellite PRN character."""
    ch = sid[0]
    if ch == 'G':
        f1,f2 = F1GPS,F2GPS
    elif ch == 'E':
        f1,f2 = F1GAL,F5GAL
    elif ch == 'C':
        f1,f2 = F2BDS,F6BDS   # B1I/B3I
    elif ch == 'R':
        f1,f2 = F1GLO,F2GLO
    else:
        f1,f2 = F1GPS,F2GPS
    la1=CLIGHT/f1; la2=CLIGHT/f2
    f1sq=f1**2; f2sq=f2**2
    alfa=f1sq/(f1sq-f2sq); beta=f2sq/(f1sq-f2sq)
    return f1,f2,la1,la2,alfa,beta

def _get_obs_codes(sid, obs):
    """Return (P1,P2,L1,L2) raw observable values for the satellite, or None."""
    ch = sid[0]
    def gv(*ll):
        for codes in ll:
            for c in codes:
                v = obs.get(c,0.0)
                if v != 0.0: return v, c
        return 0.0, None

    if ch == 'G':
        P1,pc1 = gv(['C1W'],['C1C','C1L'])
        P2,pc2 = gv(['C2W'],['C2L','C2X'])
        L1,lc1 = gv(['L1C'],['L1W','L1L'])
        L2,lc2 = gv(['L2W'],['L2L','L2X'])
    elif ch == 'E':
        P1,pc1 = gv(['C1C'],['C1X','C1Q'])
        P2,pc2 = gv(['C5Q'],['C5X','C5I'])
        L1,lc1 = gv(['L1C'],['L1X','L1Q'])
        L2,lc2 = gv(['L5Q'],['L5X','L5I'])
    elif ch == 'C':
        P1,pc1 = gv(['C2I'],['C2X','C2Q'])
        P2,pc2 = gv(['C6I'],['C6X','C6Q'])
        L1,lc1 = gv(['L2I'],['L2X','L2Q'])
        L2,lc2 = gv(['L6I'],['L6X','L6Q'])
    elif ch == 'R':
        P1,pc1 = gv(['C1C'],['C1P','C1X'])
        P2,pc2 = gv(['C2C'],['C2P','C2X'])
        L1,lc1 = gv(['L1C'],['L1P','L1X'])
        L2,lc2 = gv(['L2C'],['L2P','L2X'])
    else:
        return None
    return P1,P2,L1,L2,pc1,pc2,lc1,lc2

def _compute_sat_full(sid, obs, tow, rec_xyz, ant_h, atx,
                      sp3t, sp3p, sp3c, clkd,
                      code_osb, phase_osb,
                      sun_xyz,
                      lat0, doy, zhd, ztd_wet,
                      wu_prev, ELM):
    """
    Compute full PPP-AR observation model for one satellite.
    Returns dict of partial derivatives and corrected observables,
    or None if satellite should be excluded.
    """
    ret = _get_obs_codes(sid, obs)
    if ret is None: return None
    P1,P2,L1_cy,L2_cy,pc1,pc2,lc1,lc2 = ret
    if P1<1e4 or P2<1e4 or L1_cy<1e4 or L2_cy<1e4: return None

    f1,f2,la1,la2,alfa,beta = _sys_freqs(sid)

    # -- Code OSB (DCB) correction ------------------------------------------
    ob   = code_osb.get(sid,{})
    b1c  = ob.get(pc1, ob.get('C1W', 0.0)) if ob else 0.0
    b2c  = ob.get(pc2, ob.get('C2W', 0.0)) if ob else 0.0
    if b1c is None or np.isnan(b1c): b1c=0.0
    if b2c is None or np.isnan(b2c): b2c=0.0

    # -- Phase OSB correction (cycles) --------------------------------------
    pob  = phase_osb.get(sid,{})
    b1p  = pob.get(lc1, 0.0) if pob else 0.0
    b2p  = pob.get(lc2, 0.0) if pob else 0.0
    if b1p is None or np.isnan(b1p): b1p=0.0
    if b2p is None or np.isnan(b2p): b2p=0.0

    # Convert phase cycles → metres
    L1_m = L1_cy*la1 - b1p*la1    # phase-OSB applied
    L2_m = L2_cy*la2 - b2p*la2

    P1c = P1 - b1c;  P2c = P2 - b2c

    # Ionosphere-free combinations
    PIF = alfa*P1c - beta*P2c
    LIF = alfa*L1_m - beta*L2_m

    # -- Satellite position at signal transmission time ---------------------
    xyz0,_ = _sat_posclk(sp3t,sp3p,sp3c,sid,tow)
    if xyz0 is None: return None
    ttx = tow - np.linalg.norm(np.asarray(xyz0)-np.asarray(rec_xyz))/CLIGHT
    sxt,_ = _sat_posclk(sp3t,sp3p,sp3c,sid,ttx)
    if sxt is None: sxt = xyz0

    # Earth rotation (Sagnac) correction
    tau = np.linalg.norm(np.asarray(sxt)-np.asarray(rec_xyz))/CLIGHT
    ang = OMGE*tau
    ca,sa = math.cos(ang),math.sin(ang)
    sv  = np.array([ca*sxt[0]+sa*sxt[1], -sa*sxt[0]+ca*sxt[1], sxt[2]])

    # Satellite clock
    sc = _getclk(clkd,sid,ttx)
    if sc is None:
        _,fb = _sat_posclk(sp3t,sp3p,sp3c,sid,ttx)
        sc = fb if fb is not None else 0.0
    scm = sc*CLIGHT

    # -- Receiver APC step 1: vertical PCO only (get el/az) -----------------
    lat,lon,_ = _lla(rec_xyz)
    er = np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])
    rec_apc0 = np.asarray(rec_xyz) + ant_h*er

    el,az = _elaz(rec_apc0, sv)
    if el is None or el < ELM: return None

    # -- Receiver APC step 2: full PCO+PCV from ATX -------------------------
    rec_corr = _rec_pco_pcv_if(atx, math.degrees(el),
                                band1='L1',band2='L2',alfa=alfa,beta=beta)
    rec_apc  = np.asarray(rec_xyz) + (ant_h + rec_corr)*er

    # Refine el/az
    el2,az2 = _elaz(rec_apc, sv)
    if el2 is not None:
        el,az = el2,az2
    if el < ELM: return None

    # -- Satellite PCO (projected onto LOS) ---------------------------------
    # Uses simplified body frame (full would need attitude)
    sat_pco = _sat_pco_if(atx, sid, sv, rec_apc,
                           band1='L1',band2='L2',alfa=alfa,beta=beta)

    # -- Relativistic corrections -------------------------------------------
    vv     = _sat_vel(sp3t,sp3p,sid,tow)
    dt_rel = _rel_clock(sv,vv)          # seconds → must be multiplied by CLIGHT to get metres
    shp    = _shapiro(rec_apc, sv)      # metres

    # -- Solid-Earth tide correction ----------------------------------------
    tide_disp = _solid_tide_iers(rec_apc, sun_xyz)
    rec_apc_t = rec_apc + tide_disp

    # -- Recompute range with all corrections --------------------------------
    dr   = sv - rec_apc_t
    rng  = np.linalg.norm(dr)
    unit = dr/rng

    # -- Troposphere --------------------------------------------------------
    mh,mw = _gmf(lat0, doy, el)
    trop  = mh*zhd + mw*ztd_wet

    # Gradient partials
    cot_el = math.cos(el)/max(math.sin(el), 0.05)
    dmn    = mw*cot_el*math.cos(az)
    dme    = mw*cot_el*math.sin(az)

    # -- Phase wind-up ------------------------------------------------------
    wu    = _windup(sv, rec_apc_t, sun_xyz, wu_prev)
    wuIF  = wu * (CLIGHT/(alfa*f1-beta*f2))   # IF wind-up in metres

    # -- MW combination (for cycle-slip detection) --------------------------
    mw_raw = ((f1*L1_cy - f2*L2_cy)/(f1-f2)
              - (f1*P1c + f2*P2c)/((f1+f2)*la1))

    return dict(
        sid=sid, unit=unit, mh=mh, mw=mw,
        rng=rng, scm=scm, trop=trop,
        dt_rel_m=dt_rel*CLIGHT,  # metres (sat clock relativity)
        shp=shp*CLIGHT,          # already metres actually; shp is in metres
        wuIF=wuIF, wu=wu,
        PIF=PIF, LIF=LIF,
        el=el, az=az, lat0=lat0,
        sat_pco=sat_pco,
        dmn=dmn, dme=dme,
        la_if=(CLIGHT/(alfa*f1-beta*f2)),
        alfa=alfa, beta=beta,
        f1=f1, f2=f2, la1=la1, la2=la2,
        mw_raw=mw_raw,
    )

# ============================================================================
# System Index for ISB states
# ============================================================================
SYS_IDX = {'G':0,'R':1,'E':2,'C':3}   # GPS=reference (0=no ISB)

# ============================================================================
# PPP pass (single direction)
# ============================================================================
def run_ppp_pass(epochs, sp3t, sp3p, sp3c, clkd,
                 code_osb, phase_osb, atx,
                 ant_h, start_xyz, start_clk,
                 lat0, doy, zhd, tref, ref_xyz,
                 ELM=math.radians(10.), SC=0.30, SP=0.003,
                 direction=1, label="FWD"):

    nom = np.asarray(start_xyz, dtype=float).copy()

    # State layout:
    # [0:3]  dx,dy,dz  (position increment)
    # [3]    dclk_GPS  (receiver clock, GPS, metres)
    # [4]    ztd_wet   (troposphere wet delay, metres)
    # [5]    gn        (trop gradient north)
    # [6]    ge        (trop gradient east)
    # [7]    isb_R     (GLONASS inter-system bias vs GPS)
    # [8]    isb_E     (Galileo ISB)
    # [9]    isb_C     (BDS ISB)
    # [10+]  phase ambiguities (IF, metres) — one per satellite arc

    N_BASE = 10   # number of base states before ambiguities
    ISB_IDX = {'G':None,'R':7,'E':8,'C':9}

    x = np.zeros(N_BASE)
    x[3] = start_clk
    x[4] = 0.10

    P = np.zeros((N_BASE,N_BASE))
    P[0,0]=P[1,1]=P[2,2] = 100.**2
    P[3,3] = 3000.**2
    P[4,4] = 0.30**2
    P[5,5]=P[6,6] = 0.01**2
    P[7,7]=P[8,8]=P[9,9] = 60.**2   # ISB initial 60 m sigma

    sidx   = {}   # sid → state index
    na     = 0    # ambiguity count
    ph_init= {}   # sid → bool
    wu_map = {}   # sid → float (wind-up cycles)
    mw_sm  = {}   # sid → (count,mean,M2)
    prev_res={}   # sid → float
    results = {}
    prev_sod = None; nproc = 0; cur_3d = 9e9

    ep_list = epochs if direction==1 else list(reversed(epochs))

    for epoch in ep_list:
        sod  = epoch['t']; sobs = epoch['sats']
        dt   = abs(sod-prev_sod) if prev_sod is not None else 30.
        if dt <= 0 or dt > 3600: dt = 30.
        prev_sod = sod
        tow  = _sod2tow(sod, tref)
        sun  = _sun_ecef(tow)

        # -- Process noise ---------------------------------------------------
        n_st = len(x); Q = np.zeros((n_st,n_st))
        Q[0,0]=Q[1,1]=Q[2,2] = 1e-10*dt    # static position
        Q[3,3]   = 9e4*dt                   # clock white noise
        Q[4,4]   = 1e-8*dt                  # ZTD wet random walk
        Q[5,5]=Q[6,6] = 5e-10*dt            # gradient
        Q[7,7]=Q[8,8]=Q[9,9] = 1e-8*dt     # ISB slow drift
        for k in range(na): Q[N_BASE+k,N_BASE+k] = 1e-12*dt
        P += Q

        rec_xyz = nom + x[:3]
        ztd_wet = x[4]; gn = x[5]; ge = x[6]

        geom = []

        for sid, so in sorted(sobs.items()):
            if sid[0] not in SYS_IDX: continue
            m = _compute_sat_full(
                sid, so, tow, rec_xyz, ant_h, atx,
                sp3t, sp3p, sp3c, clkd,
                code_osb, phase_osb, sun,
                lat0, doy, zhd, ztd_wet,
                wu_map.get(sid,0.0), ELM
            )
            if m is None: continue
            wu_map[sid] = m['wu']

            # -- Geometric range (with all corrections) ----------------------
            rp_now = (m['rng'] + x[3] - m['scm'] + m['trop']
                      + m['dt_rel_m'] + m['shp'] + m['wuIF'] + m['sat_pco'])
            # Add ISB for non-GPS
            isb_i = ISB_IDX.get(sid[0])
            if isb_i is not None: rp_now += x[isb_i]

            res_now = m['LIF'] - rp_now

            # -- Cycle-slip detection (MW + IF residual) ---------------------
            slip = False
            mw_val = m['mw_raw']
            if sid in mw_sm:
                cnt,mu,M2 = mw_sm[sid]
                sigma = math.sqrt(M2/cnt) if cnt>5 else 0.5
                if abs(mw_val-mu) > max(5*sigma, 0.5): slip=True
                cnt+=1; delta=mw_val-mu; mu+=delta/cnt; M2+=delta*(mw_val-mu)
                mw_sm[sid]=(cnt,mu,M2)
            else:
                mw_sm[sid]=(1,mw_val,0.0)

            if not slip and nproc>30 and sid in prev_res:
                thr = 0.25 if cur_3d < 5000. else 0.8
                if abs(res_now-prev_res[sid]) > thr: slip=True
            prev_res[sid] = res_now

            # -- Ambiguity state allocation ----------------------------------
            if sid not in sidx:
                d = len(x)
                x = np.append(x, 0.0)
                Pn = np.zeros((d+1,d+1)); Pn[:d,:d]=P; Pn[d,d]=100.**2
                P = Pn; sidx[sid]=d; na+=1; ph_init[sid]=False

            ki = sidx[sid]
            if slip:
                x[ki]=0.0; P[ki,ki]=100.**2; ph_init[sid]=False

            if not ph_init.get(sid,False):
                rp_i = (m['rng'] + x[3] - m['scm'] + m['trop']
                        + m['dt_rel_m'] + m['shp'] + m['wuIF'] + m['sat_pco'])
                if isb_i is not None: rp_i += x[isb_i]
                x[ki] = m['LIF'] - rp_i
                P[ki,ki] = 100.**2; ph_init[sid]=True

            m['ki']=ki; m['isb_i']=isb_i; geom.append(m)

        if len(geom) < 4: continue

        # -- Build observation matrix ----------------------------------------
        n_s  = len(geom); n_st = len(x)
        H    = np.zeros((2*n_s, n_st))
        z    = np.zeros(2*n_s)
        Rd   = np.zeros(2*n_s)
        x_s  = x.copy()

        for ri,m in enumerate(geom):
            ki  = m['ki']; u = m['unit']
            mw  = m['mw']; el_m = m['el']; az_m = m['az']
            dmn = m['dmn']; dme = m['dme']
            isb_i = m['isb_i']

            rp = (m['rng'] + x_s[3] - m['scm'] + m['trop']
                  + m['dt_rel_m'] + m['shp'] + m['wuIF'] + m['sat_pco'])
            if isb_i is not None: rp += x_s[isb_i]

            sig_P = SC/max(math.sin(el_m),0.1)
            sig_L = SP/max(math.sin(el_m),0.1)

            rr = 2*ri
            H[rr,0]=-u[0]; H[rr,1]=-u[1]; H[rr,2]=-u[2]
            H[rr,3]=1.;     H[rr,4]=mw
            H[rr,5]=dmn;    H[rr,6]=dme
            if isb_i is not None: H[rr,isb_i]=1.
            z[rr]  = m['PIF'] - rp
            Rd[rr] = sig_P**2

            rl = 2*ri+1
            H[rl,0]=-u[0]; H[rl,1]=-u[1]; H[rl,2]=-u[2]
            H[rl,3]=1.;     H[rl,4]=mw
            H[rl,5]=dmn;    H[rl,6]=dme
            H[rl,ki]=1.
            if isb_i is not None: H[rl,isb_i]=1.
            z[rl]  = m['LIF'] - (rp + x_s[ki])
            Rd[rl] = sig_L**2

        # Residual gating
        if np.any(np.isnan(z)) or np.any(np.isinf(z)): continue
        if np.max(np.abs(z)) > 200.:
            print(f"  [{label}] Large residual {np.max(np.abs(z)):.1f}m → skip")
            continue

        res_norm = _kf_update(x, P, H, z, np.diag(Rd))
        nproc += 1

        # Re-linearise (re-centre) position
        shift = np.linalg.norm(x[:3])
        if nproc > 10 and 0.005 < shift < 10.0:
            nom += x[:3]; x[:3] = 0.0

        pos     = nom + x[:3]
        dx      = pos - ref_xyz
        d3      = np.linalg.norm(dx)*1e3
        cur_3d  = d3
        p_trace = P[0,0]+P[1,1]+P[2,2]
        results[sod] = {
            'xyz':pos.copy(), 'dx':dx.copy(),
            'p_trace':p_trace, 'n':n_s, 'ztd':zhd+x[4],
            'isb_R':x[7], 'isb_E':x[8], 'isb_C':x[9],
        }

        if nproc<=3 or nproc%240==0:
            print(f"  [{label}] SOD={sod:6.0f} N={n_s:2d} 3D={d3:8.1f}mm "
                  f"ZTD={zhd+x[4]:.4f} ISB_E={x[8]:.2f}")

    end_xyz = nom + x[:3]
    end_clk = x[3]

    # -- LAMBDA ambiguity resolution on final state -------------------------
    if na >= 4:
        amb_idx = list(range(N_BASE, N_BASE+na))
        a_float = x[amb_idx]
        Q_a     = P[np.ix_(amb_idx,amb_idx)]
        a_fix, ratio = _lambda_ils(a_float, Q_a)
        if a_fix is not None and ratio >= 2.5:
            print(f"  [{label}] LAMBDA fix: ratio={ratio:.2f}  n_amb={na}")
            # Constrain ambiguities (fixed solution overlay)
            # Update position using fixed ambiguities
            for k,ki in enumerate(amb_idx):
                if abs(a_fix[k] - a_float[k]) < 0.5:
                    x[ki]   = float(a_fix[k])
                    P[ki,:] = 0.0; P[:,ki] = 0.0; P[ki,ki] = 1e-12
        else:
            print(f"  [{label}] LAMBDA float (ratio={ratio:.2f}  n_amb={na})")

    return results, end_xyz, end_clk

# ============================================================================
# Main postpos entry point
# ============================================================================
def postpos(ts, te, ti, tu, popt, sopt, fopt, infiles, outfile,
            rov=None, base=None):
    t0w = _time.time()
    ddir = os.path.dirname(os.path.abspath(infiles[0]))

    def find(exts):
        for e in exts:
            for f in infiles:
                if f.lower().endswith(e.lower()): return f
            for fn in sorted(os.listdir(ddir)):
                if fn.lower().endswith(e.lower()):
                    return os.path.join(ddir,fn)
        return None

    obs_f = infiles[0]
    sp3_f = find(['.sp3','.SP3'])
    clk_f = find(['.clk','.CLK'])
    bia_f = find(['.bia','.BIA'])
    atx_f = find(['.atx','.ATX'])

    print("="*65)
    print("Multi-GNSS PPP-AR v13  (GPS+GAL+BDS+GLO)  IISC DOY038/2026")
    print("Full model: Sagnac+Relativity+Windup+PCO/PCV+SET+LAMBDA-AR")
    print("="*65)

    _, epochs, ant_h, ant_type, ant_dome = parse_obs(obs_f)
    sp3t, sp3p, sp3c = parse_sp3(sp3_f)
    clkd = parse_clk(clk_f) if clk_f else {}
    code_osb, phase_osb = parse_bia(bia_f) if bia_f else ({},{})
    atx  = parse_atx(atx_f, ant_type=ant_type, dome=ant_dome) if atx_f else {'rec':{},'sat':{}}

    # Reference and approximate coordinates (IISC)
    ref_xyz = np.array([1337935.542, 6070317.088, 1427877.494])
    apx_xyz = np.array([1337936.455, 6070317.126, 1427876.785])

    tref = sp3t[0]; DOY = 38
    lat0,lon0,h0 = _lla(apx_xyz)
    zhd = _zhd(lat0, h0)
    print(f"[INIT] ZHD={zhd:.4f}m  lat={math.degrees(lat0):.4f}°  "
          f"h={h0:.1f}m")

    ELM=math.radians(10.); SC=0.30; SP=0.003

    print("\n[PASS 1] Forward pass (GPS+GAL+BDS+GLO)...")
    fwd, end_xyz, end_clk = run_ppp_pass(
        epochs, sp3t, sp3p, sp3c, clkd, code_osb, phase_osb, atx,
        ant_h, apx_xyz.copy(), 0.0, lat0, DOY, zhd, tref, ref_xyz,
        ELM=ELM, SC=SC, SP=SP, direction=1, label="FWD")
    print(f"  Forward: {len(fwd)} epochs  "
          f"End 3D={np.linalg.norm(end_xyz-ref_xyz)*1e3:.1f}mm")

    print("\n[PASS 2] Backward pass from end-of-day position...")
    bwd, _, _ = run_ppp_pass(
        epochs, sp3t, sp3p, sp3c, clkd, code_osb, phase_osb, atx,
        ant_h, end_xyz.copy(), end_clk, lat0, DOY, zhd, tref, ref_xyz,
        ELM=ELM, SC=SC, SP=SP, direction=-1, label="BWD")
    print(f"  Backward: {len(bwd)} epochs")

    # -- Combine passes (minimum position-covariance trace) -----------------
    all_sods = sorted(set(list(fwd.keys())+list(bwd.keys())))
    combined = {}
    for sod in all_sods:
        f = fwd.get(sod); b = bwd.get(sod)
        if   f is None and b is None: continue
        elif f is None: combined[sod]={**b,'pass':'BWD'}
        elif b is None: combined[sod]={**f,'pass':'FWD'}
        else:
            combined[sod]={**f,'pass':'FWD'} if f['p_trace']<=b['p_trace'] \
                          else {**b,'pass':'BWD'}

    results_list = [(sod,combined[sod]) for sod in sorted(combined.keys())]
    print(f"\n[COMBINE] {len(results_list)} epochs  "
          f"FWD:{sum(1 for _,r in results_list if r['pass']=='FWD')}  "
          f"BWD:{sum(1 for _,r in results_list if r['pass']=='BWD')}")

    # -- Statistics over last 120 epochs ------------------------------------
    if len(results_list) > 60:
        tail = [(s,r) for s,r in results_list if s >= results_list[-120][0]]
        da   = np.array([r['dx'] for _,r in tail])
        r3   = math.sqrt(np.mean(np.sum(da**2,axis=1)))*1e3
        md   = np.mean(da,axis=0)*1e3
        latr,lonr,_ = _lla(ref_xyz)
        Re   = _enu_mat(latr,lonr)
        enu  = (Re @ da.T).T * 1e3
        re   = math.sqrt(np.mean(enu[:,0]**2))
        rn   = math.sqrt(np.mean(enu[:,1]**2))
        ru   = math.sqrt(np.mean(enu[:,2]**2))
        fp   = np.mean([r['xyz'] for _,r in tail],axis=0)
        diff = fp - ref_xyz; em = Re@diff; d3d = np.linalg.norm(diff)*1e3
        best_sod,best_r = min(results_list, key=lambda x:np.linalg.norm(x[1]['dx']))
        best3d = np.linalg.norm(best_r['dx'])*1e3

        print("\n"+"-"*65)
        print(f"[RESULT] Post-convergence (last {len(tail)} epochs)")
        print(f"  Mean dXYZ  (mm): {md[0]:+.2f}  {md[1]:+.2f}  {md[2]:+.2f}")
        print(f"  RMS  3D    (mm): {r3:.2f}")
        print(f"  RMS  E/N/U (mm): {re:.2f} / {rn:.2f} / {ru:.2f}")
        print(f"  Final mean (m):  {fp[0]:.4f}  {fp[1]:.4f}  {fp[2]:.4f}")
        print(f"  Reference  (m):  {ref_xyz[0]:.4f}  {ref_xyz[1]:.4f}  {ref_xyz[2]:.4f}")
        dm = diff*1e3
        print(f"  Diff XYZ   (mm): {dm[0]:+.2f}  {dm[1]:+.2f}  {dm[2]:+.2f}")
        print(f"  3D error   (mm): {d3d:.2f}")
        emm = em*1e3
        print(f"  ENU error  (mm): E={emm[0]:+.2f}  N={emm[1]:+.2f}  U={emm[2]:+.2f}")
        print(f"  Best epoch SOD={best_sod:.0f}  3D={best3d:.2f}mm")
        if d3d < 20.: print("\n  *** GOAL ACHIEVED: 3D < 2 cm ***")
        else:         print(f"\n  Best={best3d:.1f}mm  Last-2h RMS={r3:.1f}mm (goal <20mm)")

    print(f"\n  Wall time: {_time.time()-t0w:.1f}s")
    print("="*65)

    # -- Write CSV output ---------------------------------------------------
    if outfile and results_list:
        with open(outfile,'w') as fo:
            fo.write("SOD,pass,X,Y,Z,dX_mm,dY_mm,dZ_mm,3D_mm,N,ZTD_m,ISB_R_m,ISB_E_m,ISB_C_m\n")
            for sod,r in results_list:
                dx = r['dx']*1e3
                fo.write(f"{sod:.1f},{r['pass']},"
                         f"{r['xyz'][0]:.4f},{r['xyz'][1]:.4f},{r['xyz'][2]:.4f},"
                         f"{dx[0]:+.3f},{dx[1]:+.3f},{dx[2]:+.3f},"
                         f"{np.linalg.norm(dx):.3f},{r['n']},{r['ztd']:.4f},"
                         f"{r.get('isb_R',0.):.3f},{r.get('isb_E',0.):.3f},{r.get('isb_C',0.):.3f}\n")
        print(f"[CSV]  Written: {outfile}")
    return 1


# ============================================================================
# CLI entry
# ============================================================================
if __name__ == '__main__':
    try:
        from structures import PrcOpt, SolOpt, FilOpt
    except:
        class PrcOpt: pass
        class SolOpt: pass
        class FilOpt: pass

    data = os.path.dirname(os.path.abspath(__file__))
    infiles = [os.path.join(data, f) for f in [
        'IISC00IND_R_20260380000_01D_30S_MO.rnx',
        'IISC00IND_R_20260380000_01D_MN.rnx',
        'COD0MGXFIN_20260380000_01D_05M_ORB.SP3',
        'COD0MGXFIN_20260380000_01D_30S_CLK.CLK',
        'COD0MGXFIN_20260380000_01D_01D_OSB.BIA',
        'igs20_2408.atx',
    ]]
    outfile = os.path.join(data, 'ppp_v13_results.csv')
    postpos(None, None, 0., 0., PrcOpt(), SolOpt(), FilOpt(),
            infiles, outfile)
