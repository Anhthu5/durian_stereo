import os
import cv2, numpy as np, json, re
from pathlib import Path

ROOT=Path(os.environ['DURIAN_CALIB_IMAGES'])
OUT=Path(os.environ.get('DURIAN_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data'))) / 'calibration'; OUT.mkdir(parents=True,exist_ok=True)
PATTERN=(8,6); SQUARE=1.0

def idx(p):
    m=re.search(r'_p(\d+)_',p.name); return int(m.group(1))
A={idx(p):p for p in ROOT.glob('*_camA_*.png')}
B={idx(p):p for p in ROOT.glob('*_camB_*.png')}
obj=np.zeros((PATTERN[0]*PATTERN[1],3),np.float32)
obj[:,:2]=np.mgrid[0:PATTERN[0],0:PATTERN[1]].T.reshape(-1,2)*SQUARE
objs=[]; ptsA=[]; ptsB=[]; used=[]; size=None
flags=cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY
def read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),flags)
for i in sorted(set(A)&set(B)):
    ia=read_image(A[i]); ib=read_image(B[i]); size=(ia.shape[1],ia.shape[0])
    ga=cv2.cvtColor(ia,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(ib,cv2.COLOR_BGR2GRAY)
    oka,ca=cv2.findChessboardCornersSB(ga,PATTERN,flags=flags)
    okb,cb=cv2.findChessboardCornersSB(gb,PATTERN,flags=flags)
    if oka and okb:
        # The symmetric checkerboard may be returned in the opposite order in one camera.
        # Pick the ordering whose inter-camera displacement is most spatially consistent.
        ca_grid=ca.reshape(PATTERN[1],PATTERN[0],2)
        cb_grid=cb.reshape(PATTERN[1],PATTERN[0],2)
        variants=[cb_grid,cb_grid[::-1,::-1],cb_grid[:,::-1],cb_grid[::-1,:]]
        cb_grid=min(variants,key=lambda v: float(np.mean(np.std(v-ca_grid,axis=(0,1)))))
        cb=cb_grid.reshape(-1,1,2).copy()
        objs.append(obj.copy()); ptsA.append(ca.astype(np.float32)); ptsB.append(cb.astype(np.float32)); used.append(i)
if len(used)<8: raise RuntimeError(f'Only {len(used)} valid stereo checkerboards')
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,200,1e-7)
rmsA,KA,dA,_,_=cv2.calibrateCamera(objs,ptsA,size,None,None,criteria=criteria)
rmsB,KB,dB,_,_=cv2.calibrateCamera(objs,ptsB,size,None,None,criteria=criteria)
st_flags=cv2.CALIB_FIX_INTRINSIC
def stereo_fit(indices):
    oo=[objs[j] for j in indices]; aa=[ptsA[j] for j in indices]; bb=[ptsB[j] for j in indices]
    return cv2.stereoCalibrate(oo,aa,bb,KA,dA,KB,dB,size,criteria=criteria,flags=st_flags)

# Robustly remove calibration views that are inconsistent with a single rigid stereo rig.
keep=list(range(len(used)))
fit=stereo_fit(keep)
while len(keep)>8 and fit[0]>1.5:
    trials=[]
    for j in keep:
        kk=[x for x in keep if x!=j]
        ff=stereo_fit(kk)
        trials.append((ff[0],j,ff))
    best=min(trials,key=lambda x:x[0])
    if best[0] >= fit[0]*0.97: break
    keep.remove(best[1]); fit=best[2]
rmsS,KA,dA,KB,dB,R,T,E,F=fit
used_all=used[:]
used=[used[j] for j in keep]
R1,R2,P1,P2,Q,roi1,roi2=cv2.stereoRectify(KA,dA,KB,dB,size,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
map1x,map1y=cv2.initUndistortRectifyMap(KA,dA,R1,P1,size,cv2.CV_32FC1)
map2x,map2y=cv2.initUndistortRectifyMap(KB,dB,R2,P2,size,cv2.CV_32FC1)
np.savez_compressed(OUT/'stereo_calibration.npz',KA=KA,dA=dA,KB=KB,dB=dB,R=R,T=T,E=E,F=F,R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,map1x=map1x,map1y=map1y,map2x=map2x,map2y=map2y,image_size=np.array(size),pattern=np.array(PATTERN),square_size=SQUARE)
summary={'pattern_inner_corners':PATTERN,'square_size_relative':SQUARE,'detected_pairs':used_all,'valid_pairs':used,'rejected_pairs':[x for x in used_all if x not in used],'n_valid':len(used),'image_size':size,'rms_camA':float(rmsA),'rms_camB':float(rmsB),'rms_stereo':float(rmsS),'K_A':KA.tolist(),'dist_A':dA.ravel().tolist(),'K_B':KB.tolist(),'dist_B':dB.ravel().tolist(),'R':R.tolist(),'T_relative':T.ravel().tolist(),'baseline_relative':float(np.linalg.norm(T)),'roi1':list(map(int,roi1)),'roi2':list(map(int,roi2))}
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Visual verification on first valid pair.
i=used[0]; a=read_image(A[i]); b=read_image(B[i]); ar=cv2.remap(a,map1x,map1y,cv2.INTER_LINEAR); br=cv2.remap(b,map2x,map2y,cv2.INTER_LINEAR)
canvas=np.hstack([ar,br])
for y in range(30,size[1],40): cv2.line(canvas,(0,y),(size[0]*2,y),(0,0,255),1)
cv2.imwrite(str(OUT/'rectification_check.png'),canvas)
print(json.dumps(summary,indent=2))
