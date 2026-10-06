import pandas as pd, numpy as np, json
from scipy import stats
SRC='/mnt/user-data/uploads/20260211_aingeru_palacios/20260310_datuak_tratatzeko/cor_height_results.csv'
d=pd.read_csv(SRC)
d['rise']=d.rebound_frame-d.impact_frame
d['fall']=d.impact_frame-d.initial_frame
ok=d[d.quality_flag=='ok'].copy()
out={}
G=['golf_egurra','golf_zorua','tenis_egurra','tenis_zorua']
ok['grp']=ok.ball+'_'+ok.surface

# ---- A. per cell, t-based CI
rows=[]
for (g,h),sub in ok.groupby(['grp','nominal_drop_height_m']):
    n=len(sub); m=sub.cor_height_method.mean(); s=sub.cor_height_method.std(ddof=1)
    sem=s/np.sqrt(n); tcrit=stats.t.ppf(.975,n-1)
    rows.append(dict(grp=g,h=h,n=n,mean=m,sd=s,sem=sem,ci_t=tcrit*sem,ci_196=1.96*sem,
                     h1px=sub.initial_height_px.mean(),h2px=sub.rebound_height_px.mean(),
                     rise=sub.rise.mean()))
cells=pd.DataFrame(rows).sort_values(['grp','h'])
cells.to_csv('data/cells.csv',index=False)
out['ci_inflation_mean']=float((cells.ci_t/cells.ci_196).mean())

# ---- pooled by group
pool=[]
for g,sub in ok.groupby('grp'):
    n=len(sub);m=sub.cor_height_method.mean();s=sub.cor_height_method.std(ddof=1)
    pool.append(dict(grp=g,n=n,mean=m,sd=s,sem=s/np.sqrt(n)))
out['pooled']=pool

# ---- B. slope e vs h (proper inference)
slopes={}
for g,sub in ok.groupby('grp'):
    r=stats.linregress(sub.nominal_drop_height_m,sub.cor_height_method)
    slopes[g]=dict(slope=r.slope,inter=r.intercept,r2=r.rvalue**2,p=r.pvalue,se=r.stderr)
out['slope_raw']=slopes

# ---- C. offset delta, two independent estimates
est={}
for g,sub in ok.groupby('ball'):
    r1=stats.linregress(sub.nominal_drop_height_m,sub.initial_height_px)
    r2=stats.linregress(sub.rise**2,sub.rebound_height_px)
    est[g]=dict(scale_pxm=r1.slope,inter_h1=r1.intercept,inter_h1_se=r1.intercept_stderr,
                k=r2.slope,inter_h2=r2.intercept,inter_h2_se=r2.intercept_stderr,
                fps_implied=float(np.sqrt(.5*9.81*r1.slope/r2.slope)))
r1=stats.linregress(ok.nominal_drop_height_m,ok.initial_height_px)
r2=stats.linregress(ok.rise**2,ok.rebound_height_px)
est['all']=dict(scale_pxm=r1.slope,inter_h1=r1.intercept,inter_h1_se=r1.intercept_stderr,
                k=r2.slope,inter_h2=r2.intercept,inter_h2_se=r2.intercept_stderr,
                fps_implied=float(np.sqrt(.5*9.81*r1.slope/r2.slope)),
                t_inter_h2=r2.intercept/r2.intercept_stderr)
out['offset']=est

# ---- D. sensitivity of slope to delta
sens=[]
for delta in np.arange(0,12.5,0.5):
    row={'delta':float(delta)}
    for g,sub in ok.groupby('grp'):
        e=np.sqrt((sub.rebound_height_px-delta).clip(lower=.1)/(sub.initial_height_px-delta).clip(lower=.1))
        r=stats.linregress(sub.nominal_drop_height_m,e)
        row[g+'_slope']=r.slope; row[g+'_mean']=float(e.mean())
    sens.append(row)
sens=pd.DataFrame(sens); sens.to_csv('data/sensitivity.csv',index=False)

# ---- E. precision budget: d e / d(1 px) at each nominal height
prec=[]
for h in [0.5,1.0,1.5]:
    sub=ok[ok.nominal_drop_height_m==h]
    h1=sub.initial_height_px.mean(); h2=sub.rebound_height_px.mean(); e=np.sqrt(h2/h1)
    dedpx=0.5*e*np.sqrt((1/h2)**2+(1/h1)**2)   # 1 px independent errors
    prec.append(dict(h=h,h1px=h1,h2px=h2,e=e,de_1px=dedpx,de_1px_native=dedpx/4))
prec=pd.DataFrame(prec); prec.to_csv('data/precision.csv',index=False)
out['precision']=prec.to_dict('records')

# ---- F. scatter for figures
ok[['grp','nominal_drop_height_m','cor_height_method','rise','rebound_height_px','initial_height_px']].to_csv('data/points.csv',index=False)

# ---- G. what delta makes slopes vanish
for g,sub in ok.groupby('grp'):
    best=None
    for delta in np.arange(0,12,0.1):
        e=np.sqrt((sub.rebound_height_px-delta).clip(lower=.1)/(sub.initial_height_px-delta).clip(lower=.1))
        sl=stats.linregress(sub.nominal_drop_height_m,e).slope
        if best is None or abs(sl)<abs(best[1]): best=(float(delta),float(sl))
    out.setdefault('delta_nullslope',{})[g]=best

print(json.dumps(out,indent=1,default=float))
