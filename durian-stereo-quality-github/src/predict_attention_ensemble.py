"""Predict one preprocessed stereo view with the 3-fold Attention-MIL ensemble."""
import os
import argparse, json
from pathlib import Path
import numpy as np
import torch
from train_attention_mil import Model

BASE=Path(os.environ.get('DURIAN_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data')))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sample(path,max_rois=12):
    z=np.load(path,allow_pickle=True)
    g=z['global4'].astype(np.float32)/255
    s=z['roi_small'].astype(np.float32)/255
    l=z['roi_large'].astype(np.float32)/255
    m=z['roi_meta'].astype(np.float32)
    if len(s):
        ids=np.argsort(m[:,3]+.15*m[:,4])[::-1][:max_rois]
        s,l,m=s[ids],l[ids],m[ids]
    else:
        s=np.zeros((1,32,32,4),np.float32); l=s.copy(); m=np.zeros((1,5),np.float32)
    mean=np.array([.485,.456,.406,.5],np.float32)
    std=np.array([.229,.224,.225,.25],np.float32)
    g=(g-mean)/std; s=(s-mean)/std; l=(l-mean)/std
    g=torch.from_numpy(g.transpose(2,0,1)).unsqueeze(0)
    s=torch.from_numpy(s.transpose(0,3,1,2)).unsqueeze(0)
    l=torch.from_numpy(l.transpose(0,3,1,2)).unsqueeze(0)
    m=torch.from_numpy(m).unsqueeze(0)
    mask=torch.ones((1,s.shape[1]),dtype=torch.bool)
    return [x.to(DEVICE) for x in (g,s,l,m,mask)],z

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('npz',help='Tệp đã tạo bởi preprocess_stereo.py')
    ap.add_argument('--threshold',type=float,default=.5)
    args=ap.parse_args()
    tensors,z=load_sample(args.npz)
    probs=[]; weights=[]
    for fold in (1,2,3):
        ck=torch.load(BASE/'runs'/f'fold{fold}_resnet18_attention.pt',map_location=DEVICE,weights_only=False)
        model=Model(attention=True,pretrained=False).to(DEVICE)
        model.load_state_dict(ck['model']); model.eval()
        with torch.no_grad():
            logits,a=model(*tensors)
            probs.append(float(torch.softmax(logits,1)[0,1]))
            weights.append(a[0].cpu().tolist())
    p=float(np.mean(probs))
    result={'fruit_id':str(z['fruit_id']),'position':str(z['position']),
            'probability_ripe':p,'threshold':args.threshold,
            'predicted_class':'ripe' if p>=args.threshold else 'unripe',
            'fold_probabilities':probs,'roi_attention_by_fold':weights,
            'note':'Default threshold 0.5; validate calibration on held-out fruits.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
