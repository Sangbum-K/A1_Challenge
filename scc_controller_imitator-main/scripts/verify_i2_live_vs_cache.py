"""live i2_mem (closed_loop 경로) vs cached i2_mem (학습 분포) 일치 검증.

OLD closed_loop:  seg=combined_v1, ego_mot=real  → cache 와 불일치?
NEW (fix):        seg=fx640_v3,    ego_mot=zeros → cache 와 일치?
cached i2_mem 과 cosine/MSE 비교로 정량화.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
import timm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2Model, A2ModelMultiFrame
from src.model.image_models.iy_model import yaw_shift_tokens_batch

ROI_Y0, ROI_Y1 = 235, 485
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local"))
SEG = {"fx640_v3": "outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt",
       "combined_v1": "data/ckpts/seg_pretrain_combined_v1_best_miou_backbone.pt"}
A2CK = "data/ckpts/a2_v7b_d128_alpha075_best_loss.pt"


def parse_key(k): r,t=k.rsplit("__t",1); return r,int(t)
def load_roi(rb,tr,pol,run,tk):
    p=Path(rb)/tr/pol/run/"CAMERA_1"/f"{tk}_Intensity.jpeg"
    if not p.exists(): return None
    im=np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy() if im.shape[0]>=ROI_Y1 else im
def norm(roi): return torch.from_numpy(((roi.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))
def load_bb(ckpt,dev):
    bb=timm.create_model("tf_efficientnetv2_s",pretrained=False,features_only=True,out_indices=[-1])
    sd=torch.load(ckpt,map_location="cpu",weights_only=False); own=bb.state_dict()
    for k,v in sd.items():
        k2=k.replace("backbone.","")
        if k2 in own and own[k2].shape==v.shape: own[k2]=v
    bb.load_state_dict(own); return bb.to(dev).eval()


# ============================================================================
# mf20 (A2ModelMultiFrame n_frames=21) — closed_loop HOTFIX 정합 검증.
#   build_i2_cache_mf20.py 와 동일 규약: prev_offsets[5..100] tick raw token, yaw_shift=off, ego_motion=0.
#   closed_loop live 경로(backbone raw fc + deque offset fetch + forward_tokens)를 재현해
#   i2_mem_mf20 cache 와 cos/L1 비교. cos≈1 이면 정합 (성공 기준 >0.99).
MF20_OFFSETS = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
MF20_CKPT = "data/ckpts/a2_mf20_v2s_n20.pt"


def verify_mf20(ds, n, dev):
    mf=json.load(open(f"{D}/manifest_i1cache_{ds}.json")); track,pol,raw=mf["track"],mf["policy"],mf["raw_base"]
    # mf20 cache (정답) + i1 prev20 cache (token fallback 규약용 key set)
    zc=np.load(f"{D}/cache/i2_mem_mf20_{ds}.npz",allow_pickle=True)
    cache_keys=zc["keys"].tolist(); i2c=zc["i2_mem"]; a2c=zc["a2_logits"]
    kc2i={k:i for i,k in enumerate(cache_keys)}
    z1=np.load(f"{D}/cache/i1_tok_cache_{ds}_prev20.npz",allow_pickle=True)
    tok_keys=z1["keys"].tolist(); tok_all=z1["tokens"]; C_b=tok_all.shape[-1]
    k2idx={k:i for i,k in enumerate(tok_keys)}
    # lead(non-zero in-lane obj) frame 우선 샘플 — perception 가 의미있는 frame 으로 검증
    inlane=a2c[:,16:26,0].astype(np.float32).max(axis=1)
    order=np.argsort(-inlane)   # obj logit 큰 순
    cand=[i for i in order.tolist() if i < len(cache_keys)][: max(n*3, 200)]
    np.random.RandomState(0).shuffle(cand); cand=cand[:n]

    n_frames=1+len(MF20_OFFSETS)
    a2=A2ModelMultiFrame(img_backbone="tf_efficientnetv2_s",d_token=256,n_heads=4,n_ch=7,d_out=128,n_frames=n_frames).to(dev).eval()
    if C_b != a2.encoder.proj.in_channels:
        import torch.nn as nn
        a2.encoder.proj = nn.Conv1d(C_b,256,kernel_size=1).to(dev)
    sda=torch.load(MF20_CKPT,map_location=dev,weights_only=False); sda=sda.get("model",sda) if isinstance(sda,dict) else sda
    own=a2.state_dict(); skip=[k for k in list(sda.keys()) if k in own and sda[k].shape!=own[k].shape]
    for k in skip: del sda[k]
    miss,unexp=a2.load_state_dict(sda,strict=False)
    print(f"[mf20 model] {MF20_CKPT}  missing={len(miss)} unexpected={len(unexp)} shape-skip={len(skip)}")
    bb=load_bb(SEG["fx640_v3"],dev)

    def prev_keys_for(ck):
        # build_i2_cache_mf20._prev_idxs 와 동일: 없으면 직전 offset, 최종 self.
        base,tk=ck.rsplit("__t",1); tick=int(tk); out=[]; last=ck
        for off in MF20_OFFSETS:
            pk=f"{base}__t{tick-off:07d}"
            pk = pk if pk in k2idx else last
            out.append(pk); last=pk
        return out

    # Mode A: i1 cache 토큰 직접 → forward_tokens (model-wiring 정합. builder 와 동일 입력)
    # Mode B: raw image → backbone → forward_tokens (closed_loop live 전체 경로 재현)
    simsA=[];l1A=[]; simsB=[];l1B=[]; simsLg=[]
    for i in cand:
        ck=cache_keys[i]
        if ck not in k2idx: continue
        pks=prev_keys_for(ck)
        ref=i2c[i].astype(np.float32); refl=a2c[i].astype(np.float32)
        # --- Mode A: i1 cache tokens ---
        with torch.no_grad():
            tlA=[torch.from_numpy(tok_all[k2idx[ck]].astype(np.float32))[None].to(dev)]
            for pk in pks: tlA.append(torch.from_numpy(tok_all[k2idx[pk]].astype(np.float32))[None].to(dev))
            ez=torch.zeros(1,2,device=dev)
            lgA,memA=a2.forward_tokens(tlA,ez,return_mem=True)
            mA=memA[0].cpu().numpy()
        csA=float((mA.flatten()@ref.flatten())/(np.linalg.norm(mA)*np.linalg.norm(ref)+1e-9))
        simsA.append(csA); l1A.append(np.abs(mA-ref).mean())
        # --- Mode B: raw image → backbone (closed_loop live) ---
        def bbtok(key):
            roi=load_roi(raw,track,pol,*parse_key(key))
            if roi is None: return None
            with torch.no_grad():
                t=F.adaptive_avg_pool2d(bb(norm(roi).unsqueeze(0).to(dev))[-1],(1,40)).squeeze(2).transpose(1,2)
            return t
        tc=bbtok(ck)
        if tc is None: continue
        with torch.no_grad():
            tlB=[tc]; last=tc
            for pk in pks:
                t=bbtok(pk);
                if t is None: t=last
                tlB.append(t); last=t
            lgB,memB=a2.forward_tokens(tlB,ez,return_mem=True)
            mB=memB[0].cpu().numpy(); lgBn=lgB[0].cpu().numpy()
        csB=float((mB.flatten()@ref.flatten())/(np.linalg.norm(mB)*np.linalg.norm(ref)+1e-9))
        simsB.append(csB); l1B.append(np.abs(mB-ref).mean())
        # a2_logits 정합 (live vs cache)
        csLg=float((lgBn.flatten()@refl.flatten())/(np.linalg.norm(lgBn)*np.linalg.norm(refl)+1e-9))
        simsLg.append(csLg)

    print(f"\n[{ds}] mf20 verify  ({len(simsB)}/{n} frames, lead-우선 샘플)\n")
    print(f"  {'경로':50s} cos_sim   |L1 diff|")
    print(f"  {'A) i1cache tok → forward_tokens (builder 동일)':50s} {np.mean(simsA):+.4f}   {np.mean(l1A):.4f}")
    print(f"  {'B) raw image → backbone → forward_tokens (live)':50s} {np.mean(simsB):+.4f}   {np.mean(l1B):.4f}")
    print(f"  {'   a2_logits (live B vs cache)':50s} {np.mean(simsLg):+.4f}")
    print(f"\n성공 기준: live(B) i2_mem cos_sim > 0.99 → closed_loop mf20 경로가 학습 cache 와 정합.")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ds",default="sangam_lead_0520"); ap.add_argument("--n",type=int,default=24)
    ap.add_argument("--mf20",action="store_true",help="mf20(21f) closed_loop HOTFIX 정합 검증")
    args=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"
    if args.mf20:
        verify_mf20(args.ds, args.n, dev); return
    mf=json.load(open(f"{D}/manifest_i1cache_{args.ds}.json")); track,pol,raw=mf["track"],mf["policy"],mf["raw_base"]
    pb=np.load(f"{D}/prebake/f1_prebake_iy_{args.ds}_v1_brakeaug_prev20.npz",allow_pickle=True)
    ck=pb["curr_keys"].tolist(); pk=pb["prev_keys"].tolist(); dy=pb["d_yaws"]; eg=pb["ego_5d"]
    zi=np.load(f"{D}/cache/i2_mem_cache_{args.ds}_v7b_prev20.npz",allow_pickle=True)
    k2i={k:i for i,k in enumerate(zi["keys"].tolist())}; i2c=zi["i2_mem"]
    nz=(np.abs(i2c.astype(np.float32)).sum(axis=(1,2))>1e-6)
    cand=[i for i,k in enumerate(ck) if k2i.get(k) is not None and nz[k2i[k]]]
    np.random.RandomState(0).shuffle(cand); cand=cand[:args.n]
    a2=A2Model(img_backbone="tf_efficientnetv2_s",d_token=256,n_heads=4,n_ch=7,d_out=128).to(dev).eval()
    sda=torch.load(A2CK,map_location=dev,weights_only=False); sda=sda.get("model",sda) if isinstance(sda,dict) else sda
    a2.load_state_dict({k:v for k,v in sda.items() if k in a2.state_dict() and a2.state_dict()[k].shape==v.shape},strict=False)
    bbs={n:load_bb(p,dev) for n,p in SEG.items()}

    def live_i2(seg_name, ego_mode):
        sims=[]; cached=[]
        for i in cand:
            rc=load_roi(raw,track,pol,*parse_key(ck[i])); rp=load_roi(raw,track,pol,*parse_key(pk[i]))
            if rc is None or rp is None: continue
            bb=bbs[seg_name]
            ic=norm(rc).unsqueeze(0).to(dev); ip=norm(rp).unsqueeze(0).to(dev)
            with torch.no_grad():
                fc=F.adaptive_avg_pool2d(bb(ic)[-1],(1,40)).squeeze(2).transpose(1,2)
                fp=F.adaptive_avg_pool2d(bb(ip)[-1],(1,40)).squeeze(2).transpose(1,2)
                fpr=yaw_shift_tokens_batch(fp,torch.tensor([float(dy[i])],device=dev),fx=torch.tensor([640.0],device=dev))
                tcp=a2.encoder.proj(fc.transpose(1,2)).transpose(1,2); tpp=a2.encoder.proj(fpr.transpose(1,2)).transpose(1,2)
                em = torch.zeros(1,2,device=dev) if ego_mode=="zeros" else torch.tensor([[float(eg[i][0]),float(eg[i][4])]],device=dev)
                i2=a2.fc_proj(a2.fusion(tcp,tpp,em))[0].cpu().numpy()
            ref=i2c[k2i[ck[i]]].astype(np.float32)
            cs=float((i2.flatten()@ref.flatten())/(np.linalg.norm(i2)*np.linalg.norm(ref)+1e-9))
            sims.append(cs); cached.append(np.abs(i2-ref).mean())
        return np.mean(sims), np.mean(cached)

    print(f"[{args.ds}] {len(cand)} non-zero lead frames, cached i2_mem 과 비교\n")
    print(f"  {'경로':40s} cos_sim   |L1 diff|")
    for seg,ego,tag in [("combined_v1","real","OLD closed_loop (combined_v1 + ego_real)"),
                        ("fx640_v3","real","fx640_v3 + ego_real"),
                        ("combined_v1","zeros","combined_v1 + ego_zeros"),
                        ("fx640_v3","zeros","NEW fix (fx640_v3 + ego_zeros)")]:
        cs,l1=live_i2(seg,ego); print(f"  {tag:40s} {cs:+.4f}   {l1:.4f}",flush=True)
    print("\ncos_sim≈1 & L1≈0 이면 학습 분포와 일치. OLD 가 낮고 NEW 가 높으면 → 옛 closed-loop perception 불일치 확정.")


if __name__=="__main__":
    main()
