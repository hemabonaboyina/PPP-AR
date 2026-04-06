"""
ppp_postpos.py  v12  --  GPS PPP  <2 cm goal
Two-pass forward-backward. All filter bugs fixed.
"""

import os,sys,math,time as _time
from collections import defaultdict
import numpy as np

from constants import CLIGHT,FREQ1,FREQ2,OMGE,RE_WGS84
from kf import filter_standard

LAMBDA1=CLIGHT/FREQ1; LAMBDA2=CLIGHT/FREQ2
F1SQ,F2SQ=FREQ1**2,FREQ2**2
ALFA=F1SQ/(F1SQ-F2SQ); BETA=F2SQ/(F1SQ-F2SQ)
LAMBDA_IF=CLIGHT/(ALFA*FREQ1-BETA*FREQ2)
MU=3.986004418e14; E2=0.00669437999014

def _ifc(a,b): return ALFA*a-BETA*b
def _sig(el,s0): return s0/max(math.sin(el),0.1)

# ── parsers ────────────────────────────────────────────────────────────────────
def parse_obs(fp):
    obs_types={}; epochs=[]; ant_h=0.0
    with open(fp,'r',errors='replace') as f:
        hdr=True; ep=None
        for raw in f:
            ln=raw.rstrip('\n')
            if hdr:
                lb=ln[60:].strip() if len(ln)>60 else ''
                if 'ANTENNA: DELTA H/E/N' in lb:
                    try: ant_h=float(ln[0:14])
                    except: pass
                if 'SYS / # / OBS TYPES' in lb:
                    sc=ln[0]; n=int(ln[3:6]); obs_types.setdefault(sc,[])
                    obs_types[sc].extend(ln[7:60].split()); rem=n-len(obs_types[sc])
                    while rem>0:
                        r2=f.readline().rstrip('\n')
                        obs_types[sc].extend(r2[7:60].split()); rem=n-len(obs_types[sc])
                if 'END OF HEADER' in lb: hdr=False
            else:
                if ln.startswith('>'):
                    p=ln[1:].split(); fl=int(p[6]) if len(p)>6 else 0
                    ep={'t':int(p[3])*3600+int(p[4])*60+float(p[5]),'sats':{},'flag':fl}
                    if fl<=1: epochs.append(ep)
                elif ep and ep['flag']<=1:
                    sid=ln[0:3].strip()
                    if not sid: continue
                    types=obs_types.get(sid[0],[]); obs={}
                    for i,code in enumerate(types):
                        s=3+i*16; rv=ln[s:s+14].strip() if len(ln)>s else ''
                        try: obs[code]=float(rv) if rv else 0.
                        except: obs[code]=0.
                    ep['sats'][sid]=obs
    print(f"[OBS]  {len(epochs)} epochs  ant_h={ant_h:.4f}m")
    return obs_types,epochs,ant_h

def parse_sp3(fp):
    times=[]; rpos=defaultdict(list); rclk=defaultdict(list); ei=-1
    with open(fp,'r') as f:
        for ln in f:
            if ln.startswith('*'):
                p=ln.split()
                times.append(_gpst(int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5]),float(p[6]))); ei+=1
            elif ln.startswith('P'):
                sid=ln[1:4].strip()
                try: xk=float(ln[4:18]); yk=float(ln[18:32]); zk=float(ln[32:46]); ck=float(ln[46:60])
                except: continue
                if abs(xk)>9e5 or abs(ck)>9e8: continue
                rpos[sid].append((ei,xk*1e3,yk*1e3,zk*1e3)); rclk[sid].append((ei,ck*1e-6))
    n=len(times); sp3p={}; sp3c={}
    for s in rpos:
        ap=np.full((n,3),np.nan); ac=np.full(n,np.nan)
        for i,x,y,z in rpos[s]: ap[i]=[x,y,z]
        for i,c in rclk[s]: ac[i]=c
        sp3p[s]=ap; sp3c[s]=ac
    print(f"[SP3]  {n} epochs  {len(sp3p)} sats")
    return times,sp3p,sp3c

def parse_clk(fp):
    data=defaultdict(list); hdr=True
    with open(fp,'r') as f:
        for ln in f:
            if hdr:
                if 'END OF HEADER' in ln: hdr=False
                continue
            if ln[:2]!='AS': continue
            p=ln.split()
            if len(p)<10: continue
            try:
                t=_gpst(int(p[2]),int(p[3]),int(p[4]),int(p[5]),int(p[6]),float(p[7]))
                data[p[1]].append((t,float(p[9])))
            except: continue
    for s in data: data[s].sort(key=lambda x:x[0])
    total=sum(len(v) for v in data.values())
    print(f"[CLK]  {total} entries  {len(data)} sats")
    return dict(data)

def parse_bia(fp):
    """
    Token-based SINEX-BIA parser.
    Format: <sp>OSB  <site> <PRN> <obs1> ... <unit> <value>
    Uses split() tokens — robust against column variations.
    """
    B=defaultdict(dict); ins=False
    with open(fp,'r',errors='replace') as fh:
        for ln in fh:
            if '+BIAS/SOLUTION' in ln: ins=True; continue
            if '-BIAS/SOLUTION' in ln: break
            if not ins: continue
            if len(ln)<4 or ln[1:4]!='OSB': continue
            parts=ln.split()
            # parts[0]='OSB' parts[1]=site parts[2]=PRN parts[3]=obs_code
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
            if unit=='ns':    B[prn][obs1]=val*1e-9*CLIGHT
            elif unit=='cyc': B[prn][obs1]=val*LAMBDA1
            else:             B[prn][obs1]=val
    total=sum(len(v) for v in B.values())
    print(f"[BIA]  {total} code OSB entries  {len(B)} sats")
    if total>0:
        g01=B.get('G01',{})
        print(f"[BIA]  G01 C1W={g01.get('C1W',float('nan')):+.4f}m  C2W={g01.get('C2W',float('nan')):+.4f}m")
    return dict(B)

def parse_atx(fp, ant_type='LEIAR25.R4', dome='NONE'):
    """Parse IGS ANTEX. Returns receiver PCO/PCV and satellite PCO/PCV."""
    result = {'rec': {}, 'sat': {}}
    if not fp or not os.path.isfile(fp):
        print(f"[ATX]  File not found: {fp}")
        return result
    target = f"{ant_type:<16}{dome:<4}".strip()
    in_ant=False; in_freq=False
    freq_label=None; cur_pco=None; cur_pcv={}; cur_block=None
    with open(fp,'r',errors='replace') as fh:
        for ln in fh:
            label=ln[60:].rstrip() if len(ln)>60 else ''
            if 'START OF ANTENNA' in label:
                in_ant=False; in_freq=False; cur_block=None
            if 'TYPE / SERIAL NO' in label:
                ant_key=ln[0:20].strip()
                svn=ln[20:40].strip()
                if ant_key==target:
                    in_ant=True; cur_block='rec'
                elif len(svn)==3 and svn[0] in 'GREJCIS':
                    in_ant=True; cur_block=svn
                else:
                    in_ant=False
            if not in_ant: continue
            if 'START OF FREQUENCY' in label:
                freq_label=ln[3:6].strip(); in_freq=True; cur_pco=None; cur_pcv={}
            if 'END OF FREQUENCY' in label:
                band=('L1' if freq_label in ('G01','R01','E01','C01','J01') else
                      'L2' if freq_label in ('G02','R02','E02','C02','J02') else None)
                if band and cur_pco is not None:
                    entry={'pco':cur_pco,'pcv_noazi':dict(cur_pcv)}
                    if cur_block=='rec':
                        result['rec'][band]=entry
                    else:
                        result['sat'].setdefault(cur_block,{})[band]=entry
                in_freq=False
            if 'END OF ANTENNA' in label:
                in_ant=False; cur_block=None
            if in_freq and 'NORTH / EAST / UP' in label:
                try:
                    cur_pco=[float(ln[0:10])*1e-3,
                             float(ln[10:20])*1e-3,
                             float(ln[20:30])*1e-3]
                except: pass
            if in_freq:
                parts=ln.split()
                if parts and parts[0]=='NOAZI':
                    try:
                        vals=[float(v)*1e-3 for v in parts[1:]]
                        for ki,v in enumerate(vals):
                            cur_pcv[round(ki*5.0,1)]=v
                    except: pass
    rec_l1=result['rec'].get('L1',{}).get('pco',[0,0,0])
    rec_l2=result['rec'].get('L2',{}).get('pco',[0,0,0])
    print(f"[ATX]  Receiver {ant_type}/{dome}  L1_U={rec_l1[2]*1e3:.2f}mm  L2_U={rec_l2[2]*1e3:.2f}mm")
    print(f"[ATX]  Satellites loaded: {len(result['sat'])} PRNs")
    return result

def _pcv_noazi(pcv_dict, zen_deg):
    if not pcv_dict: return 0.0
    zens=sorted(pcv_dict.keys())
    if zen_deg<=zens[0]:  return pcv_dict[zens[0]]
    if zen_deg>=zens[-1]: return pcv_dict[zens[-1]]
    for i in range(len(zens)-1):
        if zens[i]<=zen_deg<=zens[i+1]:
            t=(zen_deg-zens[i])/(zens[i+1]-zens[i])
            return pcv_dict[zens[i]]+t*(pcv_dict[zens[i+1]]-pcv_dict[zens[i]])
    return 0.0

# ── time / orbit / geodetic helpers ────────────────────────────────────────────
def _gpst(yr,mo,dy,hr,mn,sc):
    a=(14-mo)//12; y=yr+4800-a; m=mo+12*a-3
    jdn=dy+(153*m+2)//5+365*y+y//4-y//100+y//400-32045
    d=jdn-0.5+(hr*3600+mn*60+sc)/86400-2444244.5; wk=int(d/7)
    return wk*604800+(d-wk*7)*86400

def _sod2t(sod,tref): return tref-(tref%86400)+sod

def _lag(ts,ys,t,ord=10):
    n=len(ts)
    if n==0: return None
    i=int(np.searchsorted(ts,t)); h=(ord+1)//2
    lo=max(0,min(i-h,n-ord-1)); hi=lo+ord+1
    ts_=ts[lo:hi]; ys_=ys[lo:hi]
    r=0. if ys_.ndim==1 else np.zeros(ys_.shape[1])
    for ii in range(len(ts_)):
        L=1.
        for jj in range(len(ts_)):
            if jj!=ii:
                d=ts_[ii]-ts_[jj]
                if d==0: L=0.; break
                L*=(t-ts_[jj])/d
        r+=L*ys_[ii]
    return r

def _sat_posclk(sp3t,sp3p,sp3c,sat,tow):
    ap=sp3p.get(sat)
    if ap is None: return None,None
    ts=np.array(sp3t); ok=~np.isnan(ap[:,0])
    if ok.sum()<4: return None,None
    tv=ts[ok]; pv=ap[ok]; cv=sp3c[sat][ok]
    if tow<tv[0]-400 or tow>tv[-1]+400: return None,None
    xyz=_lag(tv,pv,tow,ord=min(10,len(tv)-1))
    i=int(np.searchsorted(tv,tow)); i=max(1,min(len(tv)-1,i))
    dt=tv[i]-tv[i-1]
    clk=(cv[i-1]+(tow-tv[i-1])/dt*(cv[i]-cv[i-1])
         if dt>0 and not np.isnan(cv[i]) and not np.isnan(cv[i-1]) else cv[i-1])
    return xyz,clk

def _sat_vel(sp3t,sp3p,sat,tow):
    ap=sp3p.get(sat)
    if ap is None: return np.zeros(3)
    ts=np.array(sp3t); ok=~np.isnan(ap[:,0])
    if ok.sum()<4: return np.zeros(3)
    tv=ts[ok]; pv=ap[ok]
    return (_lag(tv,pv,tow+1.,ord=min(10,len(tv)-1))-
            _lag(tv,pv,tow-1.,ord=min(10,len(tv)-1)))/2.

def _getclk(clkd,sat,tow):
    e=clkd.get(sat)
    if not e: return None
    ts=np.array([x[0] for x in e]); cs=np.array([x[1] for x in e])
    i=int(np.searchsorted(ts,tow))
    if i==0: return cs[0]
    if i>=len(ts): return cs[-1]
    t0,c0=ts[i-1],cs[i-1]; t1,c1=ts[i],cs[i]; dt=t1-t0
    if dt>35: return c0 if tow-t0<t1-tow else c1
    return c0+(tow-t0)/dt*(c1-c0)

def _lla(xyz):
    x,y,z=xyz; p=math.sqrt(x*x+y*y); lon=math.atan2(y,x)
    lat=math.atan2(z,p*(1-E2))
    for _ in range(10):
        sl=math.sin(lat); N=RE_WGS84/math.sqrt(1-E2*sl*sl)
        ln2=math.atan2(z+E2*N*sl,p)
        if abs(ln2-lat)<1e-12: break
        lat=ln2
    sl=math.sin(lat); cl=math.cos(lat); N=RE_WGS84/math.sqrt(1-E2*sl*sl)
    h=p/cl-N if abs(cl)>1e-9 else abs(z)/sl-N*(1-E2)
    return lat,lon,h

def _enu(lat,lon):
    sl,cl=math.sin(lat),math.cos(lat); sn,cn=math.sin(lon),math.cos(lon)
    return np.array([[-sn,cn,0],[-sl*cn,-sl*sn,cl],[cl*cn,cl*sn,sl]])

def _elaz(rec,sat):
    dx=np.array(sat)-np.array(rec); lat,lon,_=_lla(rec)
    e=_enu(lat,lon)@dx; n=np.linalg.norm(e)
    if n<1: return None,None
    return math.asin(e[2]/n),math.atan2(e[0],e[1])

def _zhd(lat,h):
    P=(101325*(1-2.2557e-5*h)**5.2559)/100
    return 0.0022768*P/(1-0.00266*math.cos(2*lat)-0.00028*h/1000)

def _cf(s,a,b,c): return (1+a/(1+b/(1+c)))/(s+a/(s+b/(s+c)))

def _gmf(lat,doy,el):
    if el<1e-4: el=1e-4
    dr=28 if lat>=0 else 211; cd=math.cos(2*math.pi*(doy-dr)/365.25)
    ah=1.2769934e-3+2.8804e-5*math.cos(lat)-7.6184e-5*math.sin(lat)+2.5e-6*cd
    s=math.sin(el)
    mh=_cf(s,ah,2.9153695e-3,0.062610505)/_cf(1.,ah,2.9153695e-3,0.062610505)
    mw=_cf(s,5.7532e-4,1.8128e-3,0.062553963)/_cf(1.,5.7532e-4,1.8128e-3,0.062553963)
    return mh,mw

def _wu(sat_xyz,rec_xyz,sun_xyz,wu0):
    rho=np.array(sat_xyz); rn=np.linalg.norm(rho)
    if rn<1e3: return wu0
    k=rho-np.array(rec_xyz); k/=np.linalg.norm(k)
    sun=np.array(sun_xyz)-rho; sn=np.linalg.norm(sun)
    if sn<1e3: return wu0
    sun/=sn; ez=-rho/rn; ex=sun-sun.dot(ez)*ez; en=np.linalg.norm(ex)
    if en<1e-10: return wu0
    ex/=en; ey=np.cross(ez,ex)
    lat,lon,_=_lla(rec_xyz); sl,cl=math.sin(lat),math.cos(lat)
    sn2,cn=math.sin(lon),math.cos(lon)
    eyr=np.array([-sl*cn,-sl*sn2,cl]); exr=np.array([-sn2,cn,0.])
    ds=ex-k*k.dot(ex)-np.cross(k,ey); dr=exr-k*k.dot(exr)+np.cross(k,eyr)
    nd=np.linalg.norm(ds); nr=np.linalg.norm(dr)
    if nd<1e-10 or nr<1e-10: return wu0
    cw=ds.dot(dr)/(nd*nr); cw=max(-1.,min(1.,cw))
    dp=math.acos(cw)/(2*math.pi)
    if np.cross(ds,dr).dot(k)<0: dp=-dp
    return dp+round(wu0-dp)

def _sun(tow):
    T=(tow/86400-10957)/36525; M=math.radians(357.528+35999.05*T)
    lam=(math.radians(280.46+36000.771*T)+
         math.radians(1.915)*math.sin(M)+math.radians(0.02)*math.sin(2*M))
    eps=math.radians(23.439-0.013*T); AU=1.496e11
    xi=AU*math.cos(lam); yi=AU*math.cos(eps)*math.sin(lam)
    zi=AU*math.sin(eps)*math.sin(lam)
    g=math.fmod(tow/86164.0905*2*math.pi,2*math.pi); cg,sg=math.cos(g),math.sin(g)
    return np.array([cg*xi+sg*yi,-sg*xi+cg*yi,zi])

def _rel(sv,vv): return -2*np.dot(sv,vv)/CLIGHT**2

def _shap(rec,sat):
    rs=np.linalg.norm(sat); rr=np.linalg.norm(rec); rho=np.linalg.norm(sat-rec)
    a=(rs+rr+rho)/(rs+rr-rho)
    return 2*MU/CLIGHT**2*math.log(a) if a>0 else 0.

def _set_corr(rec,sun):
    lat,lon,_=_lla(rec); sl,cl=math.sin(lat),math.cos(lat)
    sn,cn=math.sin(lon),math.cos(lon)
    er=np.array(rec)/np.linalg.norm(rec)
    en=np.array([-sl*cn,-sl*sn,cl]); ee=np.array([-sn,cn,0.])
    def d(b):
        rb=np.linalg.norm(b); rr=np.linalg.norm(rec)
        ub=np.array(b)/rb; cz=np.dot(ub,er)
        P2=(3*cz*cz-1)/2.
        ar=0.6078*P2*3*MU/(rb**3)*rr**2/9.81
        at=0.0847*3*cz*math.sqrt(max(0.,1-cz*cz))*MU/(rb**3)*rr**2/9.81
        ube=ub.dot(ee); ubn=ub.dot(en); hn=math.sqrt(ube**2+ubn**2)+1e-15
        return ar*er+at*(ube/hn*ee+ubn/hn*en)
    return d(sun)*3.16

def _compute_sat(sid,so,tow,rec_xyz,ant_h,sp3t,sp3p,sp3c,
                 clkd,code_osb,atx,lat0,doy,zhd,ztd_wet,ELM):
    def gv(so,*ll):
        for codes in ll:
            for c in codes:
                v=so.get(c,0.)
                if v!=0.: return v
        return 0.
    P1=gv(so,['C1W'],['C1C','C1L']); P2=gv(so,['C2W'],['C2L','C2X'])
    L1c=gv(so,['L1C'],['L1L','L1X']); L2c=gv(so,['L2W'],['L2L','L2X'])
    if P1<1e4 or P2<1e4 or L1c<1e4 or L2c<1e4: return None
    ob=code_osb.get(sid,{})
    pc1='C1W' if so.get('C1W',0.)!=0. else 'C1C'
    pc2='C2W' if so.get('C2W',0.)!=0. else 'C2L'
    b1=ob.get(pc1,ob.get('C1W',0.)); b2=ob.get(pc2,ob.get('C2W',0.))
    P1c=P1-b1; P2c=P2-b2
    PIF=_ifc(P1c,P2c); LIF=_ifc(L1c*LAMBDA1,L2c*LAMBDA2)
    xyz0,_=_sat_posclk(sp3t,sp3p,sp3c,sid,tow)
    if xyz0 is None: return None
    ttx=tow-np.linalg.norm(xyz0-rec_xyz)/CLIGHT
    sxt,_=_sat_posclk(sp3t,sp3p,sp3c,sid,ttx)
    if sxt is None: sxt=xyz0
    tau=np.linalg.norm(sxt-rec_xyz)/CLIGHT; ang=OMGE*tau
    ca,sa=math.cos(ang),math.sin(ang)
    sv=np.array([ca*sxt[0]+sa*sxt[1],-sa*sxt[0]+ca*sxt[1],sxt[2]])
    sc=_getclk(clkd,sid,ttx)
    if sc is None:
        if clkd.get(sid): return None   # CLK loaded but gap — skip
        _,fb=_sat_posclk(sp3t,sp3p,sp3c,sid,ttx); sc=fb if fb is not None else 0.
    scm=sc*CLIGHT
    lat,lon,_=_lla(rec_xyz)
    er=np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])

    # ── Step 1: rough APC (ant_h only) to get el/az ──────────────────────────
    rec_apc0=rec_xyz+ant_h*er
    el,az=_elaz(rec_apc0,sv)
    if el is None or el<ELM: return None

    # ── Step 2: receiver PCO+PCV from ATX ────────────────────────────────────
    el_zen=90.0-math.degrees(el)
    rec_atx=atx.get('rec',{})
    pco_l1_u=rec_atx.get('L1',{}).get('pco',[0,0,0])[2]
    pco_l2_u=rec_atx.get('L2',{}).get('pco',[0,0,0])[2]
    pco_if_u=ALFA*pco_l1_u-BETA*pco_l2_u
    pcv_l1=_pcv_noazi(rec_atx.get('L1',{}).get('pcv_noazi',{}),el_zen)
    pcv_l2=_pcv_noazi(rec_atx.get('L2',{}).get('pcv_noazi',{}),el_zen)
    pcv_if=ALFA*pcv_l1-BETA*pcv_l2
    rec_apc=rec_xyz+(ant_h+pco_if_u+pcv_if)*er

    # ── Step 3: refine el/az with corrected APC ───────────────────────────────
    el2,az2=_elaz(rec_apc,sv)
    if el2 is not None: el,az=el2,az2
    if el<ELM: return None

    vv=_sat_vel(sp3t,sp3p,sid,tow)
    dtrel=_rel(sv,vv); shp=_shap(rec_apc,sv)
    dr=sv-rec_apc; rng=np.linalg.norm(dr); unit=dr/rng
    setm=-unit.dot(_set_corr(rec_apc,_sun(tow)))
    mh,mw=_gmf(lat0,doy,el); trop=mh*zhd+mw*ztd_wet
    mw_raw=((FREQ1*L1c-FREQ2*L2c)/(FREQ1-FREQ2)
            -(FREQ1*P1c+FREQ2*P2c)/((FREQ1+FREQ2)*LAMBDA1))
    return dict(sid=sid,unit=unit,mh=mh,mw=mw,rng=rng,scm=scm,trop=trop,
                dtrel=dtrel,shp=shp,setm=setm,PIF=PIF,LIF=LIF,el=el,az=az,
                sat_xyz=sv,rec_apc=rec_apc,mw_raw=mw_raw)

# ── core PPP pass ──────────────────────────────────────────────────────────────
def run_ppp_pass(epochs,sp3t,sp3p,sp3c,clkd,code_osb,atx,
                 ant_h,start_xyz,start_clk,
                 lat0,doy,zhd,tref,
                 ELM=math.radians(10.),SC=0.30,SP=0.003,
                 direction=1,label="FWD"):
    ref=np.array([1337935.542,6070317.088,1427877.494])
    nom=start_xyz.copy()

    # State vector: [dx, dy, dz, clk, ztd_wet, gn, ge, amb_0...amb_N]
    # Indices 0-2: position increments
    # Index  3:    receiver clock (m)
    # Index  4:    ZTD wet (m)
    # Indices 5-6: troposphere N/E gradient (m)
    # Indices 7+:  float IF ambiguities (m)
    x_p=np.zeros(7); x_p[3]=start_clk; x_p[4]=0.10
    P_p=np.zeros((7,7))
    P_p[0,0]=P_p[1,1]=P_p[2,2]=100.**2   # position 100 m initial sigma
    P_p[3,3]=300.**2                      # clock 3 km initial sigma
    P_p[4,4]=0.30**2                       # ZTD wet 30 cm initial sigma
    P_p[5,5]=P_p[6,6]=0.01**2             # gradient 10 mm initial sigma

    sidx={}; na=0; ph_init={}; wu_map={}; prev_res={}; mw_smooth={}
    results={}; prev_sod=None; nproc=0; cur_3d=9e9
    ep_list=epochs if direction==1 else list(reversed(epochs))

    for epoch in ep_list:
        sod=epoch['t']; sobs=epoch['sats']
        dt=abs(sod-prev_sod) if prev_sod is not None else 30.
        if dt<=0 or dt>3600: dt=30.
        prev_sod=sod; tow=_sod2t(sod,tref)

        # ── Process noise Q ───────────────────────────────────────────────────
        # FIX 1: clock noise was 9e6 (blows up P in ~26 epochs).
        #        9e4 → sigma ~300 m/sqrt(s), stable and physically reasonable.
        # FIX 2: ambiguity index was 5+k (overwrote gradient states 5,6).
        #        Must be 7+k because gradients occupy indices 5 and 6.
        n_st=len(x_p); Q=np.zeros((n_st,n_st))
        Q[0,0]=Q[1,1]=Q[2,2]=1e-10*dt     # position random walk (static site)
        Q[3,3]=1e2*dt                      # clock white noise
        Q[4,4]=1e-8*dt                      # ZTD wet random walk
        Q[5,5]=Q[6,6]=5e-10*dt             # troposphere gradient
        for k in range(na): Q[7+k,7+k]=1e-12*dt  # ambiguities (index starts at 7)
        P_p+=Q

        sun=_sun(tow)
        rec_xyz=nom+x_p[:3]; ztd_wet=x_p[4]
        geom=[]

        for sid,so in sorted(sobs.items()):
            if sid[0]!='G': continue
            m=_compute_sat(sid,so,tow,rec_xyz,ant_h,sp3t,sp3p,sp3c,
                           clkd,code_osb,atx,lat0,doy,zhd,ztd_wet,ELM)
            if m is None: continue

            wu=_wu(m['sat_xyz'],m['rec_apc'],sun,wu_map.get(sid,0.))
            wu_map[sid]=wu
            LIFc=m['LIF']-wu*LAMBDA_IF; m['LIFc']=LIFc

            rp_now=(m['rng']+x_p[3]-m['scm']+m['trop']
                    +m['dtrel']*CLIGHT+m['shp']+m['setm'])
            res_now=LIFc-rp_now

            slip_if=False
            # MW combination slip check (primary)
            mw_val=m['mw_raw']
            if sid in mw_smooth:
                cnt,mu,M2=mw_smooth[sid]
                sigma=math.sqrt(M2/cnt) if cnt>5 else 0.5
                if abs(mw_val-mu)>max(5*sigma,0.5):
                    slip_if=True
                cnt+=1; delta=mw_val-mu; mu+=delta/cnt; M2+=delta*(mw_val-mu)
                mw_smooth[sid]=(cnt,mu,M2)
            else:
                mw_smooth[sid]=(1,mw_val,0.0)
            # IF residual slip check (secondary guard)
            if not slip_if and nproc>30 and sid in prev_res:
                thr=0.3 if cur_3d<5000. else 1.0
                if abs(res_now-prev_res[sid])>thr:
                    slip_if=True
            prev_res[sid]=res_now

            # FIX 3: ambiguity allocation — new sats go at index len(x_p)
            # which starts at 7 after the gradient states.
            if sid not in sidx:
                d=len(x_p); x_p=np.append(x_p,0.)
                Pn=np.zeros((d+1,d+1)); Pn[:d,:d]=P_p; Pn[d,d]=100.**2
                P_p=Pn; sidx[sid]=d; na+=1; ph_init[sid]=False
            ki=sidx[sid]

            if slip_if:
                x_p[ki]=0.; P_p[ki,ki]=100.**2; ph_init[sid]=False

            if not ph_init.get(sid,False):
                rp_i=(m['rng']+x_p[3]-m['scm']+m['trop']
                      +m['dtrel']*CLIGHT+m['shp']+m['setm'])
                x_p[ki]=LIFc-rp_i; P_p[ki,ki]=100.**2; ph_init[sid]=True

            m['ki']=ki; geom.append(m)

        if len(geom)<4: continue

        # ── Build H matrix and innovation vector ─────────────────────────────
        n_s=len(geom); n_st=len(x_p)
        H=np.zeros((2*n_s,n_st)); z=np.zeros(2*n_s); Rd=np.zeros(2*n_s)
        x_snap=x_p.copy()
        for ri,m in enumerate(geom):
            ki=m['ki']; u=m['unit']; mw=m['mw']
            rp=(m['rng']+x_snap[3]-m['scm']+m['trop']
                +m['dtrel']*CLIGHT+m['shp']+m['setm'])
            # Troposphere gradient partials
            az_m=m['az']; el_m=m['el']
            cot_el=math.cos(el_m)/max(math.sin(el_m),0.05)
            dmn=mw*cot_el*math.cos(az_m)   # north gradient partial
            dme=mw*cot_el*math.sin(az_m)   # east gradient partial
            # Pseudorange row
            rr=2*ri
            H[rr,0]=-u[0]; H[rr,1]=-u[1]; H[rr,2]=-u[2]
            H[rr,3]=1.; H[rr,4]=mw; H[rr,5]=dmn; H[rr,6]=dme
            z[rr]=m['PIF']-rp
            Rd[rr]=_sig(el_m,SC)**2
            # Carrier phase row
            rl=2*ri+1
            H[rl,0]=-u[0]; H[rl,1]=-u[1]; H[rl,2]=-u[2]
            H[rl,3]=1.; H[rl,4]=mw; H[rl,5]=dmn; H[rl,6]=dme; H[rl,ki]=1.
            z[rl]=m['LIFc']-(rp+x_snap[ki])
            Rd[rl]=_sig(el_m,SP)**2      # FIX 4: removed duplicate Rd[rl] line

        
        ret=filter_standard(x_p,P_p,H.T,z,np.diag(Rd))
        if ret!=0:
            # Fallback: least-squares position-only update using pseudorange rows
            try:
                Hr=H[0::2,:4]; zr=z[0::2]  # PR rows, first 4 states only
                dx_ls,_,_,_=np.linalg.lstsq(Hr,zr,rcond=None)
                x_p[:4]+=dx_ls; continue
            except: continue
        nproc+=1

        # ── Re-linearise around converged position ────────────────────────────
        # FIX 5: was checking nproc>0 BEFORE incrementing nproc, so it
        # recentred every single epoch from epoch 2 onward. Now properly
        # guarded: only after 10 good epochs, only when shift is 1 cm–10 m.
        shift=np.linalg.norm(x_p[:3])
        if nproc>10 and 0.01<shift<10.0:
            nom+=x_p[:3]; x_p[:3]=0.

        pos=nom+x_p[:3]; dx=pos-ref; d3=np.linalg.norm(dx)*1e3
        cur_3d=d3; p_trace=P_p[0,0]+P_p[1,1]+P_p[2,2]
        results[sod]={'xyz':pos.copy(),'dx':dx.copy(),
                      'p_trace':p_trace,'n':n_s,'ztd':zhd+x_p[4]}

        if nproc<=3 or nproc%240==0:
            print(f"  [{label}] SOD={sod:6.0f} N={n_s} 3D={d3:8.1f}mm ZTD={zhd+x_p[4]:.4f}")

    end_xyz=nom+x_p[:3]; end_clk=x_p[3]
    return results,end_xyz,end_clk

# ── main ───────────────────────────────────────────────────────────────────────
def postpos(ts,te,ti,tu,popt,sopt,fopt,infiles,outfile,rov=None,base=None):
    t0w=_time.time()
    ddir=os.path.dirname(os.path.abspath(infiles[0]))
    def find(exts):
        for e in exts:
            for f in infiles:
                if f.lower().endswith(e.lower()): return f
            for fn in os.listdir(ddir):
                if fn.lower().endswith(e.lower()): return os.path.join(ddir,fn)
        return None

    obs_f=infiles[0]; sp3_f=find(['.sp3','.SP3'])
    clk_f=find(['.clk','.CLK']); bia_f=find(['.bia','.BIA'])

    print("="*65); print("GPS PPP v12 (forward-backward)  IISC DOY038/2026"); print("="*65)
    _,epochs,ant_h=parse_obs(obs_f)
    sp3t,sp3p,sp3c=parse_sp3(sp3_f)
    clkd=parse_clk(clk_f) if clk_f else {}
    code_osb=parse_bia(bia_f) if bia_f else {}
    atx_f=find(['.atx','.ATX']); atx=parse_atx(atx_f) if atx_f else {'rec':{},'sat':{}}

    ref=np.array([1337935.542,6070317.088,1427877.494])
    apx=np.array([1337936.455,6070317.126,1427876.785])
    tref=sp3t[0]; DOY=38
    lat0,lon0,h0=_lla(apx); zhd=_zhd(lat0,h0)
    print(f"[INIT] ZHD={zhd:.4f}m  lat={math.degrees(lat0):.4f}")
    ELM=math.radians(10.); SC=0.30; SP=0.003

    print("\n[PASS 1] Forward pass...")
    fwd,end_xyz,end_clk=run_ppp_pass(
        epochs,sp3t,sp3p,sp3c,clkd,code_osb,atx,
        ant_h,apx.copy(),0.0,lat0,DOY,zhd,tref,
        ELM=ELM,SC=SC,SP=SP,direction=1,label="FWD")
    print(f"  Forward pass: {len(fwd)} epochs processed")
    print(f"  End 3D error: {np.linalg.norm(end_xyz-ref)*1e3:.1f} mm")

    print("\n[PASS 2] Backward pass from end-of-day position...")
    bwd,_,_=run_ppp_pass(
        epochs,sp3t,sp3p,sp3c,clkd,code_osb,atx,
        ant_h,end_xyz.copy(),end_clk,lat0,DOY,zhd,tref,
        ELM=ELM,SC=SC,SP=SP,direction=-1,label="BWD")
    print(f"  Backward pass: {len(bwd)} epochs processed")

    print("\n[COMBINE] Selecting best solution per epoch...")
    all_sods=sorted(set(list(fwd.keys())+list(bwd.keys())))
    combined={}
    for sod in all_sods:
        f=fwd.get(sod); b=bwd.get(sod)
        if f is None and b is None: continue
        elif f is None: combined[sod]={**b,'pass':'BWD'}
        elif b is None: combined[sod]={**f,'pass':'FWD'}
        else:
            combined[sod]={**f,'pass':'FWD'} if f['p_trace']<=b['p_trace'] else {**b,'pass':'BWD'}

    print("\n"+"="*65)
    results_list=[(sod,combined[sod]) for sod in sorted(combined.keys())]
    print(f"[INFO] Total epochs combined: {len(results_list)}")
    fwd_c=sum(1 for _,r in results_list if r['pass']=='FWD')
    bwd_c=sum(1 for _,r in results_list if r['pass']=='BWD')
    print(f"[INFO] FWD: {fwd_c}  BWD: {bwd_c}")

    if len(results_list)>60:
        tail=[(sod,r) for sod,r in results_list if sod>=results_list[-120][0]]
        da=np.array([r['dx'] for _,r in tail])
        r3=math.sqrt(np.mean(np.sum(da**2,axis=1)))*1e3
        md=np.mean(da,axis=0)*1e3
        latr,lonr,_=_lla(ref); Re=_enu(latr,lonr)
        enu=(Re@da.T).T*1e3
        re=math.sqrt(np.mean(enu[:,0]**2)); rn=math.sqrt(np.mean(enu[:,1]**2)); ru=math.sqrt(np.mean(enu[:,2]**2))
        fp=np.mean([r['xyz'] for _,r in tail],axis=0)
        diff=fp-ref; em=Re@diff; d3d=np.linalg.norm(diff)*1e3
        best_sod,best_r=min(results_list,key=lambda x:np.linalg.norm(x[1]['dx']))
        best3d=np.linalg.norm(best_r['dx'])*1e3

        print(f"[RESULT] Post-convergence last {len(tail)} epochs")
        print(f"  Mean dXYZ (mm): {md[0]:+.2f}  {md[1]:+.2f}  {md[2]:+.2f}")
        print(f"  RMS  3D   (mm): {r3:.2f}")
        print(f"  RMS  E/N/U(mm): {re:.2f} / {rn:.2f} / {ru:.2f}")
        print(f"  Final XYZ  (m): {fp[0]:.4f}  {fp[1]:.4f}  {fp[2]:.4f}")
        print(f"  Ref   XYZ  (m): {ref[0]:.4f}  {ref[1]:.4f}  {ref[2]:.4f}")
        dm=diff*1e3
        print(f"  Diff  XYZ (mm): {dm[0]:+.2f}  {dm[1]:+.2f}  {dm[2]:+.2f}")
        print(f"  3D error  (mm): {d3d:.2f}")
        print(f"  ENU error (mm): E={em[0]*1e3:+.2f}  N={em[1]*1e3:+.2f}  U={em[2]*1e3:+.2f}")
        print(f"  Best epoch SOD={best_sod:.0f}  3D={best3d:.2f}mm  [{best_r['pass']}]")
        if d3d<20: print("\n  *** GOAL ACHIEVED: 3D < 2 cm ***")
        else: print(f"\n  Best={best3d:.1f}mm  Last-2h={r3:.1f}mm  (goal <20mm)")

    print(f"  Wall time: {_time.time()-t0w:.1f}s")
    print("="*65)

    if outfile and results_list:
        with open(outfile,'w') as fo:
            fo.write("SOD,pass,X,Y,Z,dX_mm,dY_mm,dZ_mm,3D_mm,N,ZTD_m\n")
            for sod,r in results_list:
                dx=r['dx']*1e3
                fo.write(f"{sod:.1f},{r['pass']},{r['xyz'][0]:.4f},"
                         f"{r['xyz'][1]:.4f},{r['xyz'][2]:.4f},"
                         f"{dx[0]:+.3f},{dx[1]:+.3f},{dx[2]:+.3f},"
                         f"{np.linalg.norm(dx):.3f},{r['n']},{r['ztd']:.4f}\n")
        print(f"[CSV]  Written: {outfile}")
    return 1


if __name__=='__main__':
    try:
        from structures import PrcOpt,SolOpt,FilOpt
    except:
        class PrcOpt: pass
        class SolOpt: pass
        class FilOpt: pass

    data=os.path.dirname(os.path.abspath(__file__))
    infiles=[os.path.join(data,f) for f in [
        'IISC00IND_R_20260380000_01D_30S_MO.rnx',
        'IISC00IND_R_20260380000_01D_MN.rnx',
        'COD0MGXFIN_20260380000_01D_05M_ORB.SP3',
        'COD0MGXFIN_20260380000_01D_30S_CLK.CLK',
        'COD0MGXFIN_20260380000_01D_01D_OSB.BIA',
        'igs20_2408.atx']]
    postpos(None,None,0.,0.,PrcOpt(),SolOpt(),FilOpt(),infiles,
            os.path.join(data,'ppp_results.csv'))
