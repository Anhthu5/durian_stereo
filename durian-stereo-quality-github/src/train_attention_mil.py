import os
import json, random, copy, math, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

BASE=Path(os.environ.get('DURIAN_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'data'))); DATA=BASE/'processed'; RUN=BASE/'runs'; RUN.mkdir(exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILES=sorted(DATA.glob('*.npz'))

def info(path):
    z=np.load(path,allow_pickle=True)
    return str(z['fruit_id']),int(z['class_id'])
META=[info(p) for p in FILES]
fruits=sorted(set(x[0] for x in META)); fruit_label={f:next(y for ff,y in META if ff==f) for f in fruits}

class DurianSet(Dataset):
    def __init__(self, files, train=False, max_rois=12): self.files=files; self.train=train; self.max_rois=max_rois
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        z=np.load(self.files[i],allow_pickle=True)
        g=z['global4'].astype(np.float32)/255
        rs=z['roi_small'].astype(np.float32)/255; rl=z['roi_large'].astype(np.float32)/255; meta=z['roi_meta'].astype(np.float32)
        k=min(len(rs),self.max_rois)
        if k:
            score=meta[:,3]+.15*meta[:,4]
            ids=np.argsort(score)[::-1][:k]
            if self.train and k>8:
                # Mild ROI dropout prevents relying on one accidental region.
                keep=np.random.rand(k)>.08
                if keep.sum()<4:keep[:4]=True
                ids=ids[keep]
            rs,rl,meta=rs[ids],rl[ids],meta[ids]
        else:
            rs=np.zeros((1,32,32,4),np.float32);rl=rs.copy();meta=np.zeros((1,5),np.float32)
        if self.train and random.random()<.5:
            g=np.ascontiguousarray(g[:,::-1]); rs=np.ascontiguousarray(rs[:,:,::-1]); rl=np.ascontiguousarray(rl[:,:,::-1]); meta[:,0]=1-meta[:,0]
        if self.train:
            # The same mild photometric transform is applied to RGB in global
            # and ROI tensors; geometry/depth remains aligned and biologically plausible.
            gain=random.uniform(.88,1.12); bias=random.uniform(-.05,.05)
            g[...,:3]=np.clip(g[...,:3]*gain+bias,0,1)
            rs[...,:3]=np.clip(rs[...,:3]*gain+bias,0,1); rl[...,:3]=np.clip(rl[...,:3]*gain+bias,0,1)
            if random.random()<.35:
                noise=np.random.normal(0,.012,g[...,3].shape).astype(np.float32)
                g[...,3]=np.clip(g[...,3]+noise,0,1)
                rs[...,3]=np.clip(rs[...,3]+np.random.normal(0,.012,rs[...,3].shape),0,1)
                rl[...,3]=np.clip(rl[...,3]+np.random.normal(0,.012,rl[...,3].shape),0,1)
        # Normalize RGB around ImageNet statistics; depth/topography to [-1,1].
        mean=np.array([.485,.456,.406,.5],np.float32); std=np.array([.229,.224,.225,.25],np.float32)
        g=(g-mean)/std; rs=(rs-mean)/std; rl=(rl-mean)/std
        return torch.from_numpy(g.transpose(2,0,1).copy()),torch.from_numpy(rs.transpose(0,3,1,2).copy()),torch.from_numpy(rl.transpose(0,3,1,2).copy()),torch.from_numpy(meta.copy()),int(z['class_id']),str(z['fruit_id']),str(z['position'])

def collate(batch):
    gs,ss,ll,mm,yy,ff,pp=zip(*batch); K=max(x.shape[0] for x in ss); B=len(batch)
    S=torch.zeros(B,K,4,32,32); L=torch.zeros_like(S); M=torch.zeros(B,K,5); mask=torch.zeros(B,K,dtype=torch.bool)
    for i,(s,l,m) in enumerate(zip(ss,ll,mm)):k=s.shape[0];S[i,:k]=s;L[i,:k]=l;M[i,:k]=m;mask[i,:k]=True
    return torch.stack(gs),S,L,M,mask,torch.tensor(yy),list(ff),list(pp)

class ROIEncoder(nn.Module):
    def __init__(self,out=128):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));self.fc=nn.Linear(96,out)
    def forward(self,x):return self.fc(self.net(x).flatten(1))

class Model(nn.Module):
    def __init__(self,attention=True,pretrained=True,backbone='resnet18',gated=True):
        super().__init__(); self.attention=attention
        self.backbone=backbone
        if backbone=='efficientnet_b0':
            try:net=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            except Exception:net=efficientnet_b0(weights=None)
            old=net.features[0][0]; new=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
            with torch.no_grad(): new.weight[:,:3]=old.weight;new.weight[:,3:4]=old.weight.mean(1,keepdim=True)
            net.features[0][0]=new; self.global_net=nn.Sequential(net.features,net.avgpool); global_dim=1280
        else:
            try:net=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            except Exception:net=resnet18(weights=None)
            old=net.conv1; net.conv1=nn.Conv2d(4,64,7,2,3,bias=False)
            with torch.no_grad(): net.conv1.weight[:,:3]=old.weight; net.conv1.weight[:,3:4]=old.weight.mean(1,keepdim=True)
            self.global_net=nn.Sequential(*list(net.children())[:-1]); global_dim=512
        self.roi=ROIEncoder(128); self.meta=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,32))
        self.att_v=nn.Sequential(nn.Linear(160,96),nn.Tanh())
        self.att_u=nn.Sequential(nn.Linear(160,96),nn.Sigmoid()) if gated else None
        self.att_out=nn.Linear(96,1)
        self.head=nn.Sequential(nn.Linear(global_dim+160,256),nn.ReLU(),nn.Dropout(.45),nn.Linear(256,2))
    def forward(self,g,s,l,m,mask):
        B,K=s.shape[:2]; gf=self.global_net(g).flatten(1)
        sf=self.roi(s.reshape(B*K,4,32,32)); lf=self.roi(l.reshape(B*K,4,32,32)); rf=.5*(sf+lf)
        rf=torch.cat([rf,self.meta(m.reshape(B*K,5))],1).reshape(B,K,-1)
        if self.attention:
            h=self.att_v(rf); h=h*self.att_u(rf) if self.att_u is not None else h
            score=self.att_out(h).squeeze(-1).masked_fill(~mask,-1e4); a=torch.softmax(score,1)
        else:a=mask.float()/mask.float().sum(1,keepdim=True).clamp_min(1)
        bag=(rf*a.unsqueeze(-1)).sum(1); return self.head(torch.cat([gf,bag],1)),a

def fruit_metrics(rows):
    bag=defaultdict(list); truth={}
    for f,y,p in rows:bag[f].append(p);truth[f]=y
    fs=sorted(bag); prob=np.array([np.mean(bag[f]) for f in fs]); yt=np.array([truth[f] for f in fs]); yp=(prob>=.5).astype(int)
    out={'n_fruits':len(fs),'accuracy':accuracy_score(yt,yp),'balanced_accuracy':balanced_accuracy_score(yt,yp),'macro_f1':f1_score(yt,yp,average='macro',zero_division=0),'precision_ripe':precision_score(yt,yp,zero_division=0),'recall_ripe':recall_score(yt,yp,zero_division=0),'confusion_matrix':confusion_matrix(yt,yp,labels=[0,1]).tolist()}
    try:out['roc_auc']=roc_auc_score(yt,prob)
    except:out['roc_auc']=None
    out['predictions']=[{'fruit_id':f,'true':int(truth[f]),'prob_ripe':float(np.mean(bag[f])),'pred':int(np.mean(bag[f])>=.5)} for f in fs]
    return out

def run_fold(fold,train_f,val_f,attention=True,epochs=18,backbone='resnet18'):
    tr_all=[p for p,(f,y) in zip(FILES,META) if f in train_f]; va=[p for p,(f,y) in zip(FILES,META) if f in val_f]
    # Equal number of viewing angles per fruit prevents fruits with extra captures dominating.
    byfruit=defaultdict(list)
    for p in tr_all: byfruit[info(p)[0]].append(p)
    tr=[]
    for f,ps in byfruit.items():
        ids=np.linspace(0,len(ps)-1,min(8,len(ps)),dtype=int); tr += [ps[j] for j in ids]
    ds=DurianSet(tr,True); vd=DurianSet(va,False)
    labels=[info(p)[1] for p in tr]; counts=Counter(labels); weights=[1/counts[y] for y in labels]
    tl=DataLoader(ds,batch_size=16,sampler=WeightedRandomSampler(weights,len(weights),replacement=True),collate_fn=collate,num_workers=0)
    vl=DataLoader(vd,batch_size=16,shuffle=False,collate_fn=collate,num_workers=0)
    model=Model(attention=attention,backbone=backbone,gated=True).to(DEVICE)
    # First train the task-specific ROI/fusion heads. On GPU the last global
    # backbone block is unfrozen for low-learning-rate domain adaptation.
    for p in model.global_net.parameters():p.requires_grad=False
    if DEVICE.type=='cuda':
        blocks=list(model.global_net[0].children()) if backbone=='efficientnet_b0' else list(model.global_net.children())
        for p in blocks[-1].parameters():p.requires_grad=True
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('global_net')],'lr':3e-4}]
    gp=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('global_net')]
    if gp:groups.append({'params':gp,'lr':1e-5})
    opt=torch.optim.AdamW(groups,weight_decay=2e-4)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.,1.35],device=DEVICE),label_smoothing=.04)
    amp=DEVICE.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=None; patience=5; bad=0
    for ep in range(epochs):
        model.train()
        for g,s,l,m,mask,y,f,p in tl:
            g,s,l,m,mask,y=[x.to(DEVICE) for x in (g,s,l,m,mask,y)];opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=amp):logit,a=model(g,s,l,m,mask);loss=crit(logit,y)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
        model.eval();rows=[]
        with torch.no_grad():
            for g,s,l,m,mask,y,f,p in vl:
                logit,a=model(g.to(DEVICE),s.to(DEVICE),l.to(DEVICE),m.to(DEVICE),mask.to(DEVICE));pr=torch.softmax(logit,1)[:,1].cpu().numpy()
                rows += [(ff,int(yy),float(pp)) for ff,yy,pp in zip(f,y.numpy(),pr)]
        met=fruit_metrics(rows);score=met['macro_f1']
        if best is None or score>best['metrics']['macro_f1']+1e-6:
            best={'epoch':ep+1,'metrics':met,'state':copy.deepcopy(model.state_dict())};bad=0
        else:bad+=1
        if bad>=patience:break
    # Save attention evidence for validation images at best state.
    model.load_state_dict(best['state']);model.eval();att_rows=[]
    with torch.no_grad():
        for g,s,l,m,mask,y,f,p in vl:
            logit,a=model(g.to(DEVICE),s.to(DEVICE),l.to(DEVICE),m.to(DEVICE),mask.to(DEVICE));pr=torch.softmax(logit,1)[:,1].cpu();a=a.cpu()
            for i in range(len(f)):
                k=int(mask[i].sum()); top=torch.topk(a[i,:k],min(5,k)) if k else None
                att_rows.append({'fruit_id':f[i],'position':p[i],'true':int(y[i]),'prob_ripe':float(pr[i]),'top_roi_indices':top.indices.tolist() if top else [],'top_attention':top.values.tolist() if top else []})
    tag=f'{backbone}_{"attention" if attention else "mean"}'
    torch.save({'model':best['state'],'fold':fold,'attention':attention,'backbone':backbone,'train_fruits':sorted(train_f),'val_fruits':sorted(val_f),'metrics':best['metrics']},RUN/f'fold{fold}_{tag}.pt')
    (RUN/f'fold{fold}_{"attention" if attention else "mean"}_attention.json').write_text(json.dumps(att_rows,indent=2),encoding='utf-8')
    return {'fold':fold,'best_epoch':best['epoch'],**best['metrics']}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['attention','mean'],default='attention');ap.add_argument('--backbone',choices=['resnet18','efficientnet_b0'],default='resnet18');ap.add_argument('--folds',type=int,default=5);ap.add_argument('--epochs',type=int,default=18);args=ap.parse_args()
    y=np.array([fruit_label[f] for f in fruits]);groups=np.array(fruits);X=np.zeros(len(fruits))
    cv=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=SEED)
    results=[]
    for fold,(tr,va) in enumerate(cv.split(X,y,groups),1):
        print('fold',fold,'train',len(tr),'val',len(va),flush=True)
        results.append(run_fold(fold,set(groups[tr]),set(groups[va]),attention=args.mode=='attention',epochs=args.epochs,backbone=args.backbone))
        print(results[-1],flush=True)
    keys=['accuracy','balanced_accuracy','macro_f1','precision_ripe','recall_ripe','roc_auc']
    summary={'mode':args.mode,'backbone':args.backbone,'device':str(DEVICE),'n_fruits':len(fruits),'n_images':len(FILES),'folds':results,'mean':{k:float(np.nanmean([r[k] for r in results if r[k] is not None])) for k in keys},'std':{k:float(np.nanstd([r[k] for r in results if r[k] is not None])) for k in keys}}
    (RUN/f'cv_{args.backbone}_{args.mode}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary['mean'],indent=2))
