"""업샘플(1-layer U-Net) 이 2프레임 vL_rel 추정에 효과 있는지 검증.
3 config (동일 minimal regressor, 2프레임 curr+prev20 → closing scalar):
  A. f5 → pool40   (현재 i_1, stride32 deep, 256ch)
  B. d4 → pool40   (up4 decode, 동일 해상도 → decode 효과만)
  C. d4 → pool80   (up4 decode + 2× 해상도 → 제안 full)
metric: val corr(pred closing, GT closing). A<B<C 면 decode/해상도 각각 기여.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.image_models.seg_pretrain_model import SegPretrainModel
D = os.environ.get("SCC_DATA_DIR") or "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
ROI=(235,485); MEAN=np.array([.485,.456,.406],np.float32); STD=np.array([.229,.224,.225],np.float32)
dev="cuda" if torch.cuda.is_available() else "cpu"

def parse(k): r,t=k.rsplit("__t",1); return r,int(t)
def load_img(rb,tr,pol,run,tk):
    p=Path(rb)/tr/pol/run/"CAMERA_1"/f"{tk}_Intensity.jpeg"
    if not p.exists(): return None
    im=np.array(Image.open(p).convert("RGB"))
    if im.shape[0]<ROI[1]: return None
    roi=im[ROI[0]:ROI[1]]
    return torch.from_numpy(((roi.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))

# seg model (encoder + up4)
seg=SegPretrainModel(backbone="tf_efficientnetv2_s").to(dev).eval()
sd=torch.load("outputs_multi/seg_pretrain_fx640_v3/best_miou_full.pt",map_location=dev,weights_only=False)
sd=sd.get("model",sd) if isinstance(sd,dict) else sd
seg.load_state_dict({k:v for k,v in sd.items() if k in seg.state_dict() and seg.state_dict()[k].shape==v.shape},strict=False)
for p in seg.parameters(): p.requires_grad_(False)

@torch.no_grad()
def feats(x):  # x (1,3,H,W) → (f5 stride32, d4 stride16)
    fs=seg.backbone(x); f5=fs[-1]; f4=fs[-2]; d4=seg.up4(f5,f4)
    return f5, d4
def pool(t,n): return F.adaptive_avg_pool2d(t,(1,n)).squeeze(2).transpose(1,2)[0].cpu()  # (n,C)

# gather lead frames (sangam_lead_0520 + 0521)
mans={}; samples=[]; framecache={}
for ds in ['sangam_lead_0520','sangam_lead_0521','DriverB_new']:
    mf=json.load(open(f"{D}/manifest_i1cache_{ds}.json")); rb,trk,pol=mf['raw_base'],mf['track'],mf['policy']
    pb=np.load((f"{D}/prebake/f1_prebake_iy_DriverB_new_logged_fixed_prev20_trajfix.npz" if ds=="DriverB_new" else f"{D}/prebake/f1_prebake_iy_{ds}_v1_brakeaug_prev20.npz"),allow_pickle=True)
    ck=pb['curr_keys'].tolist(); pk=pb['prev_keys'].tolist()
    zg=np.load(f"{D}/cache/inlane_gt_{ds}.npz",allow_pickle=True); zk={k:i for i,k in enumerate(zg['keys'].tolist())}; gd=zg['inlane_dist']; gc=zg['inlane_closing']
    seen=set()
    for i,k in enumerate(ck):
        gi=zk.get(k)
        if gi is None or not(np.isfinite(gd[gi]) and gd[gi]<60): continue
        samples.append((k,pk[i],float(gc[gi]),rb,trk,pol))
        for kk in (k,pk[i]): seen.add((kk,rb,trk,pol))
    print(f"[{ds}] samples so far {len(samples)}, unique frames {len(seen)}",flush=True)
# ★ frame 단위 dedup (curr_key 중복 제거) — 같은 frame 이 train/val 양쪽 들어가 leakage(corr 1.0) 방지
import random; random.seed(0); random.shuffle(samples)
_seen=set(); _ded=[]
for s in samples:
    if s[0] in _seen: continue
    _seen.add(s[0]); _ded.append(s)
samples=_ded[:2500]
print(f"[dedup] unique-curr samples {len(samples)} (frame-level, no leakage)",flush=True)
uniq={}
for k,pkk,_,rb,trk,pol in samples:
    for kk in (k,pkk): uniq[kk]=(rb,trk,pol)
print(f"[extract] {len(uniq)} unique frames → seg forward",flush=True)
t0=time.time()
for n,(kk,(rb,trk,pol)) in enumerate(uniq.items()):
    im=load_img(rb,trk,pol,*parse(kk))
    if im is None: continue
    f5,d4=feats(im.unsqueeze(0).to(dev))
    framecache[kk]=(pool(f5,40), pool(d4,40), pool(d4,80))  # A,B,C tokens
    if (n+1)%500==0: print(f"  {n+1}/{len(uniq)} ({time.time()-t0:.0f}s)",flush=True)
samples=[s for s in samples if s[0] in framecache and s[1] in framecache]
print(f"[data] usable samples {len(samples)}",flush=True)
GT=torch.tensor([s[2] for s in samples],dtype=torch.float32)

class TinyVrel(nn.Module):
    """readout 수정: mean-pool(width 측정 불가) → in-lane 영역 conv+flatten (해상도 보존)."""
    def __init__(self,C,n_col):
        super().__init__()
        self.proj=nn.Linear(C,64); self.attn=nn.MultiheadAttention(64,4,batch_first=True)
        self.lo=int(n_col*0.40); self.hi=int(n_col*0.65)   # in-lane region (40:16-26, 80:32-52)
        self.conv=nn.Conv1d(64,32,3,padding=1); self.act=nn.GELU()
        self.head=nn.Linear(32*(self.hi-self.lo),1)        # flatten in-lane → resolution 보존
    def forward(self,curr,prev):
        qc=self.proj(curr); kp=self.proj(prev); f,_=self.attn(qc,kp,kp)   # (B,n_col,64)
        inl=f[:,self.lo:self.hi,:].transpose(1,2)          # (B,64,n_inlane)
        h=self.act(self.conv(inl))                          # (B,32,n_inlane)
        return self.head(h.flatten(1)).squeeze(-1)

# ★ run 단위 split (인접 tick leakage 제거): run 별로 통째 train/val 배정
_runs=sorted({s[0].rsplit("__t",1)[0] for s in samples})
import random as _r; _r.Random(0).shuffle(_runs)
_va_runs=set(_runs[:max(1,len(_runs)//5)])
_TR=torch.tensor([i for i,s in enumerate(samples) if s[0].rsplit("__t",1)[0] not in _va_runs])
_VA=torch.tensor([i for i,s in enumerate(samples) if s[0].rsplit("__t",1)[0] in _va_runs])
print(f"[split] runs={len(_runs)} val_runs={len(_va_runs)} | train {len(_TR)} val {len(_VA)}",flush=True)

def run(idx_tok, C, name):
    cur=torch.stack([framecache[s[0]][idx_tok] for s in samples]); prv=torch.stack([framecache[s[1]][idx_tok] for s in samples])
    tr,va=_TR,_VA
    m=TinyVrel(C, cur.shape[1]).to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-3); gt=GT.to(dev)
    cur=cur.to(dev); prv=prv.to(dev)
    for ep in range(60):
        m.train(); bs=1024
        for s in range(0,len(tr),bs):
            b=tr[s:s+bs]; pred=m(cur[b],prv[b]); loss=F.mse_loss(pred,gt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pv=m(cur[va],prv[va]).cpu().numpy()
    g=gt[va].cpu().numpy(); cr=np.corrcoef(pv,g)[0,1]
    print(f"  {name:24s} corr(pred closing, GT)= {cr:+.3f}",flush=True); return cr

print("\n=== 2프레임 vL_rel 회귀 corr (해상도/decode 효과) ===")
A=run(0,framecache[samples[0][0]][0].shape[-1],"A f5->40 (현재 i_1)")
B=run(1,framecache[samples[0][0]][1].shape[-1],"B d4->40 (decode만)")
C=run(2,framecache[samples[0][0]][2].shape[-1],"C d4->80 (decode+해상도=제안)")
print(f"\nA→B (decode 효과): {B-A:+.3f} | B→C (해상도 효과): {C-B:+.3f} | A→C (full): {C-A:+.3f}")
print("효과 미미하면 → 업샘플 redundant (vL_rel 병목은 다른 곳: 2프레임 차분 노이즈 등)")
