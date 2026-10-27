"""Exploratory fruit-level association; cached disparity is not metric depth."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json, csv, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(os.environ.get('DURIAN_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data'))); O=B/'results'/'association_36'; O.mkdir(exist_ok=True)
names=['R_median','G_median','B_median','Lab_a_median','Lab_b_median','green_fraction','yellow_fraction','texture_neighbor_contrast','texture_neighbor_homogeneity','relative_disparity_median','relative_disparity_iqr','topo_roi_contrast','topo_center_surround','nearest_peak_distance','peak_spacing_cv','selected_peak_count','quality_sharpness','quality_valid_fraction']
def savecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
def mask(rgb):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    m=cv2.inRange(h,np.array([8,22,25],np.uint8),np.array([105,255,250],np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,l,s,_=cv2.connectedComponentsWithStats(m)
    return l==(1+np.argmax(s[1:,4])) if n>1 else m>0
def extract(p):
    with np.load(p,allow_pickle=True) as z:
        a=z['global4']; rgb=a[:,:,:3]; m=mask(rgb); h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)/255
        f=[float(np.median(rgb[:,:,j][m])/255) for j in range(3)]
        f += [float(np.median(lab[:,:,j][m])/255) for j in (1,2)]
        f += [float(np.mean(((h[:,:,0]>=lo)&(h[:,:,0]<hi)&(h[:,:,1]>40))[m])) for lo,hi in [(35,85),(15,35)]]
        validpairs=m[:,1:]&m[:,:-1]; delta=(gray[:,1:]-gray[:,:-1])[validpairs]
        f += [float(np.mean(delta**2)),float(np.mean(1/(1+delta**2)))]
        vm=cv2.resize(z['mask_valid'],(224,224),interpolation=cv2.INTER_NEAREST)>0; dv=a[:,:,3][m&vm]/255
        f += [float(np.median(dv)) if len(dv) else np.nan,float(np.ptp(np.percentile(dv,[25,75]))) if len(dv) else np.nan]
        t=z['roi_large'][:,:,:,3].astype(float)/255; meta=z['roi_meta']; k=len(meta)
        if k:
            f += [float(np.median(np.percentile(t,90,axis=(1,2))-np.percentile(t,10,axis=(1,2)))),float(np.median(np.mean(t[:,14:18,14:18],axis=(1,2))-np.median(t,axis=(1,2))))]
        else:f += [np.nan,np.nan]
        if k>1:
            yy,xx=np.where(m); xy=meta[:,:2]*224; d=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);np.fill_diagonal(d,np.inf); nn=d.min(1)/max(np.ptp(xx),np.ptp(yy),1)
            f += [float(np.median(nn)),float(nn.std()/(nn.mean()+1e-9))]
        else:f += [np.nan,np.nan]
        f += [k,float(cv2.Laplacian(gray.astype(np.float32),cv2.CV_32F)[m].var()),float(vm[m].mean())]
        return str(z['fruit_id']),np.array(f)
def met(y,p):return dict(mae=float(mean_absolute_error(y,p)),rmse=float(mean_squared_error(y,p)**.5),r2=float(r2_score(y,p)))
def models():
    return {'PLSR':(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PLSRegression(scale=False)),{'plsregression__n_components':[1,2,3]}),'SVR':(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),SVR()),{'svr__C':[1,10,100],'svr__epsilon':[.2,1.],'svr__gamma':['scale']})}
def nested(X,y,seed=42):
    out={n:np.zeros(len(y)) for n in ['mean','PLSR','SVR']}; detail=[]
    for fold,(tr,te) in enumerate(KFold(3,shuffle=True,random_state=seed).split(X)):
        out['mean'][te]=y[tr].mean()
        for n,(pipe,grid) in models().items():
            fit=GridSearchCV(pipe,grid,cv=KFold(3,shuffle=True,random_state=seed+fold+1),scoring='neg_mean_squared_error',n_jobs=1).fit(X[tr],y[tr])
            out[n][te]=fit.predict(X[te]).ravel()
            detail.append(dict(model=n,fold=fold,train=met(y[tr],fit.predict(X[tr]).ravel()),validation=met(y[te],out[n][te]),parameters=fit.best_params_))
    return out,detail
def main():
    rng=np.random.default_rng(42); labels=json.loads((B/'labels_brix_clean.json').read_text(encoding='utf-8'))['fruit_summary']; labels={x['fruit_id']:x for x in labels}; bags=defaultdict(list)
    for p in sorted((B/'processed').glob('*.npz')):
        f,x=extract(p)
        if f in labels:bags[f].append(x)
    fruits=sorted(bags); assert len(fruits)==36
    X=np.array([np.nanmedian(bags[f],axis=0) for f in fruits]); y=np.array([np.median(labels[f]['values']) for f in fruits])
    # Keep all recorded measurements; do not silently delete extreme biological values.
    rows=[dict(fruit_id=f,brix=float(y[i]),n_views=len(bags[f]),**dict(zip(names,X[i]))) for i,f in enumerate(fruits)]; savecsv(O/'fruit_features.csv',rows)
    print('extracted',len(fruits),'fruits',sum(map(len,bags.values())),'views',flush=True)
    cor=[]
    for j,n in enumerate(names):
        ok=np.isfinite(X[:,j]); xx=X[ok,j]; yy=y[ok]; rho=float(spearmanr(xx,yy).statistic)
        boots=[]
        for _ in range(2000):
            ids=rng.integers(0,len(xx),len(xx));boots.append(spearmanr(xx[ids],yy[ids]).statistic)
        xr=rankdata(xx);yr=rankdata(yy); xp=xr-xr.mean();yp=yr-yr.mean();den=np.linalg.norm(xp)*np.linalg.norm(yp)
        null=np.array([np.dot(xp,rng.permutation(yp))/den for _ in range(4999)])
        lo,hi=np.nanpercentile(boots,[2.5,97.5]);cor.append(dict(feature=n,rho=rho,ci_low=float(lo),ci_high=float(hi),p=float((1+np.sum(np.abs(null)>=abs(rho)))/5000)))
    p=np.array([r['p'] for r in cor]);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];qs=np.empty(len(p));qs[order]=np.minimum(q,1)
    for r,v in zip(cor,qs):r['q_BH']=float(v)
    savecsv(O/'correlations.csv',sorted(cor,key=lambda r:r['p']))
    print('correlations complete',flush=True)
    # Quality variables are diagnostic, never model inputs.
    xx=X[:,:16]; results={};details=[]; allpred=[]
    for seed in [42,7,77]:
        pred,d=nested(xx,y,seed);results[str(seed)]={n:met(y,v) for n,v in pred.items()};details+= [dict(seed=seed,**a) for a in d]
        allpred += [dict(seed=seed,fruit_id=f,brix=float(y[i]),**{n:float(v[i]) for n,v in pred.items()}) for i,f in enumerate(fruits)]
    savecsv(O/'out_of_fold_predictions.csv',allpred)
    original,_=nested(xx,y); nulls={n:[] for n in models()}
    for i in range(199):
        yp=rng.permutation(y);pr,_=nested(xx,yp)
        for n in nulls:nulls[n].append(met(yp,pr[n])['r2'])
        if (i+1)%20==0:print('nested permutations',i+1,'/199',flush=True)
    perms={n:dict(observed_r2=met(y,original[n])['r2'],p=float((1+sum(v>=met(y,original[n])['r2'] for v in vals))/200),null_r2=vals) for n,vals in nulls.items()}
    final={}
    import joblib
    for n,(pipe,grid) in models().items():
        fit=GridSearchCV(pipe,grid,cv=KFold(3,shuffle=True,random_state=42),scoring='neg_mean_squared_error').fit(xx,y);joblib.dump(fit.best_estimator_,O/(n+'.joblib'));final[n]=dict(parameters=fit.best_params_,training_metrics=met(y,fit.predict(xx).ravel()))
    report=dict(n_fruits=36,features=names,labels='median of all values in existing labels_brix_clean.json; source decimal/date conversion inherited, not independently revalidated',cv=results,fold_details=details,permutation=perms,final_models=final,limitations=['18_07 only; exploratory','cached channel D is normalized disparity, not metric depth','topography is clipped normalized disparity residual','peak count and spacing affected by NMS and cap32','fruit mask re-estimated at224 resolution','fruit independence assumed; tree/orchard grouping unverified','199 nested permutations; minimum p=.005'])
    (O/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,7));cs=sorted(cor,key=lambda a:a['rho']);v=np.array([a['rho'] for a in cs]);ax.errorbar(v,np.arange(len(cs)),xerr=[v-np.array([a['ci_low'] for a in cs]),np.array([a['ci_high'] for a in cs])-v],fmt='o');ax.set_yticks(range(len(cs)),[a['feature'] for a in cs]);ax.axvline(0,color='gray');ax.set_xlabel('Spearman rho, fruit bootstrap 95% CI');fig.tight_layout();fig.savefig(O/'correlation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,n in zip(axs,['PLSR','SVR']):ax.scatter(y,original[n]);ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--');ax.set(title=n,xlabel='Measured Brix',ylabel='Outer-fold prediction')
    fig.tight_layout();fig.savefig(O/'predictions.png',dpi=160)
    print(json.dumps(dict(cv=results,permutation={n:{k:v for k,v in a.items() if k!='null_r2'} for n,a in perms.items()},significant=[a for a in cor if a['q_BH']<.05]),indent=2),flush=True)
if __name__=='__main__':main()
