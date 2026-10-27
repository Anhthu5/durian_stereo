import cv2, json, math, os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE=Path(os.environ.get('DURIAN_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data')))
MANIFEST=json.loads((BASE/'stereo_manifest.json').read_text(encoding='utf-8'))
AUDIT=json.loads((BASE/'dataset_audit.json').read_text(encoding='utf-8'))
LABEL={x['fruit_id']:x for x in AUDIT['fruit_summary']}
CAL=np.load(BASE/'calibration'/'stereo_calibration.npz')
OUT=BASE/'processed'; OUT.mkdir(parents=True,exist_ok=True)
QA=BASE/'qa'; QA.mkdir(parents=True,exist_ok=True)

map1x,map1y,map2x,map2y=[CAL[x] for x in ('map1x','map1y','map2x','map2y')]

def fruit_mask(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    # Durian is green/yellow/brown and more saturated than the white curtain.
    m=cv2.inRange(hsv,np.array([8,22,25],np.uint8),np.array([105,255,250],np.uint8))
    m[:20]=0; m[:,-20:]=0; m[:,:20]=0
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))
    n,lab,stats,cents=cv2.connectedComponentsWithStats(m)
    best=0; score=-1
    h,w=m.shape
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]; cx,cy=cents[i]
        central=math.exp(-(((cx-w*.52)/(w*.45))**2+((cy-h*.53)/(h*.45))**2))
        s=area*central
        if s>score:score=s;best=i
    mask=(lab==best).astype(np.uint8)*255 if best else np.zeros_like(m)
    # Keep body, remove thin stem and unstable outer boundary for spike search later.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    return mask

def stereo_disparity(a,b):
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    minD=0; numD=160; block=5
    left=cv2.StereoSGBM_create(minDisparity=minD,numDisparities=numD,blockSize=block,
      P1=8*block*block,P2=32*block*block,disp12MaxDiff=2,uniquenessRatio=8,
      speckleWindowSize=80,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    right=cv2.ximgproc.createRightMatcher(left)
    dl=left.compute(ga,gb); dr=right.compute(gb,ga)
    wls=cv2.ximgproc.createDisparityWLSFilter(left)
    wls.setLambda(8000.0); wls.setSigmaColor(1.3)
    filt=wls.filter(dl,ga,None,dr).astype(np.float32)/16.0
    valid=(filt>1)&(filt<minD+numD-1)
    return filt,valid

def norm_u8(x,valid,lo=2,hi=98):
    vals=x[valid]
    if vals.size<50:return np.zeros(x.shape,np.uint8)
    a,b=np.percentile(vals,[lo,hi]); y=np.clip((x-a)/(b-a+1e-6),0,1)
    return (y*255).astype(np.uint8)

def find_rois(topo,mask,valid,kmax=32):
    h,w=topo.shape
    inner=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
    usable=inner&valid
    smooth=cv2.GaussianBlur(topo,(0,0),1.0)
    dil=cv2.dilate(smooth,np.ones((9,9),np.uint8))
    vals=smooth[usable]
    if vals.size<100:return []
    thr=max(24,float(np.percentile(vals,90)))
    ys,xs=np.where((smooth==dil)&usable&(smooth>=thr))
    if len(xs)==0:return []
    # confidence: topographic height + local contrast + valid neighborhood.
    cand=[]
    for x,y in zip(xs,ys):
        y0,y1=max(0,y-8),min(h,y+9); x0,x1=max(0,x-8),min(w,x+9)
        local=smooth[y0:y1,x0:x1]; vv=valid[y0:y1,x0:x1]
        contrast=float(smooth[y,x]-np.percentile(local,25))
        vr=float(vv.mean())
        conf=.65*float(smooth[y,x])/255+.25*max(0,contrast)/100+.10*vr
        cand.append((conf,int(x),int(y),vr))
    cand.sort(reverse=True)
    # Adaptive NMS based on apparent fruit width.
    # Keep representative spikes rather than densely sampling every visible tip.
    xx=np.where(inner)[1]; min_dist=int(np.clip((xx.max()-xx.min())/9 if xx.size else 30,24,42))
    chosen=[]
    for c in cand:
        if all((c[1]-q[1])**2+(c[2]-q[2])**2>=min_dist**2 for q in chosen):
            chosen.append(c)
            if len(chosen)>=kmax:break
    return chosen

def crop_pad(x,cx,cy,size):
    r=size//2
    pad=r+2
    xp=cv2.copyMakeBorder(x,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    cx,cy=int(round(cx))+pad,int(round(cy))+pad
    return xp[cy-r:cy-r+size,cx-r:cx-r+size].copy()

def process_one(item,qa=False):
    fruit=item['fruit_id']; pos=item['position']; stem=f'{fruit}_{pos}'
    target=OUT/f'{stem}.npz'
    if target.exists() and not qa:return {'ok':1,'skipped':1}
    a=cv2.imread(item['left']); b=cv2.imread(item['right'])
    ar=cv2.remap(a,map1x,map1y,cv2.INTER_LINEAR); br=cv2.remap(b,map2x,map2y,cv2.INTER_LINEAR)
    mask=fruit_mask(ar)
    disp,valid=stereo_disparity(ar,br)
    valid &= mask>0
    disp_u8=norm_u8(disp,valid)
    # Remove smooth fruit curvature; positive residual emphasizes protruding spike tips.
    bg=cv2.medianBlur(disp_u8,31)
    rel=cv2.subtract(disp_u8,bg)
    topo=norm_u8(rel,valid&((mask>0)))
    rois=find_rois(topo,mask,valid)
    if len(rois)<4:
        # fallback to geometry maxima from normalized disparity
        rois=find_rois(disp_u8,mask,valid)
    rgb=cv2.cvtColor(ar,cv2.COLOR_BGR2RGB)
    rgba=np.dstack([rgb,topo])
    small=[]; large=[]; meta=[]
    for conf,x,y,vr in rois:
        small.append(cv2.resize(crop_pad(rgba,x,y,20),(32,32),interpolation=cv2.INTER_LINEAR))
        large.append(crop_pad(rgba,x,y,32))
        meta.append([x/ar.shape[1],y/ar.shape[0],math.hypot(x-ar.shape[1]/2,y-ar.shape[0]/2)/math.hypot(ar.shape[1]/2,ar.shape[0]/2),conf,vr])
    small=np.stack(small) if small else np.zeros((0,32,32,4),np.uint8)
    large=np.stack(large) if large else np.zeros((0,32,32,4),np.uint8)
    glob_rgb=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    # Global branch uses relative disparity/depth; ROI branch uses local topography.
    glob_depth=cv2.resize(disp_u8,(224,224),interpolation=cv2.INTER_AREA)
    global4=np.dstack([glob_rgb,glob_depth]).astype(np.uint8)
    lab=LABEL[fruit]
    np.savez_compressed(target,global4=global4,roi_small=small,roi_large=large,roi_meta=np.asarray(meta,np.float32),mask_valid=np.uint8(valid),class_id=np.int64(1 if lab['class_label']=='ripe' else 0),fruit_id=fruit,position=pos,brix_mean=np.float32(lab['brix_mean'] or np.nan))
    if qa:
        ov=ar.copy();
        for j,(conf,x,y,vr) in enumerate(rois): cv2.circle(ov,(x,y),5,(0,0,255),1); cv2.putText(ov,str(j+1),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
        panels=[ar,br,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),cv2.applyColorMap(disp_u8,cv2.COLORMAP_TURBO),cv2.cvtColor(topo,cv2.COLOR_GRAY2BGR),ov]
        panels=[cv2.resize(x,(320,240)) for x in panels]
        cv2.imwrite(str(QA/f'{stem}.jpg'),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    return {'ok':1,'fruit':fruit,'position':pos,'k':len(rois),'valid_ratio':float(valid[mask>0].mean()) if np.any(mask>0) else 0}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--qa',action='store_true'); args=ap.parse_args()
    items=MANIFEST[:args.limit] if args.limit else MANIFEST
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut=[ex.submit(process_one,x,args.qa and i<12) for i,x in enumerate(items)]
        for i,f in enumerate(as_completed(fut),1):
            try:results.append(f.result())
            except Exception as e:results.append({'ok':0,'error':repr(e)})
            if i%25==0:print(i,'/',len(items),flush=True)
    good=[x for x in results if x.get('ok')]
    summary={'n':len(results),'ok':len(good),'errors':[x for x in results if not x.get('ok')],'k_mean':float(np.mean([x.get('k',0) for x in good])) if good else 0,'k_min':int(np.min([x.get('k',0) for x in good])) if good else 0,'k_max':int(np.max([x.get('k',0) for x in good])) if good else 0,'valid_ratio_mean':float(np.mean([x.get('valid_ratio',0) for x in good])) if good else 0}
    (BASE/'preprocess_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
