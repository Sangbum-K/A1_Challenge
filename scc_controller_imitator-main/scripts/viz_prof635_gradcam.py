"""Prof635SideModel (#654 frozen main + f_2 lead-gated pedal override) grad-cam / 분해 분석.

lead-TTC 샘플(inlane_dist<20 & closing>1)에서:
  image curr/prev → seg backbone(hook) → a_2 v7b → i2_mem + a2_logits
  traj6([-y,x,speed]) + ego6 + i2_mem → Prof635SideModel → steer, final_pedal
  + gate g, base_pedal(f_1), f2_pedal(override) 분해
  brake attribution: d(-final_pedal)/d(backbone conv feat) → grad-cam over image
  → f_2 가 앞차(image)를 보고 brake override 하는지 검증.
"""
from __future__ import annotations
import os, sys, argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import timm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.model.image_models.a2_model import A2Model
from src.model.image_models.iy_model import yaw_shift_tokens_batch
from src.model.image_models.prof635_side_model import Prof635SideModel
from src.datamodule.prof635_dataset import build_traj6, build_ego6

ROI_Y0, ROI_Y1 = 235, 485; ROI_H, IMG_W = 250, 1280
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
D = os.environ.get("SCC_DATA_DIR") or ("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data" if os.path.isdir("/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local")); CK = "data/ckpts"


def parse_key(k): r, t = k.rsplit("__t", 1); return r, int(t)
def load_roi(rb, tr, pol, run, tk):
    p = Path(rb)/tr/pol/run/"CAMERA_1"/f"{tk}_Intensity.jpeg"
    if not p.exists(): return None
    im = np.array(Image.open(p).convert("RGB"))
    return im[ROI_Y0:ROI_Y1].copy() if im.shape[0] >= ROI_Y1 else im
def norm(roi): return torch.from_numpy(((roi.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="sangam_lead_0520")
    ap.add_argument("--prebake", default=f"{D}/prebake/f1_prebake_iy_sangam_lead_0520_v1_brakeaug_prev20.npz")
    ap.add_argument("--inlane-gt", default=f"{D}/cache/inlane_gt_sangam_lead_0520.npz")
    ap.add_argument("--manifest", default=f"{D}/manifest_i1cache_sangam_lead_0520.json")
    ap.add_argument("--model-ckpt", default="outputs_multi/prof635_side_v3_654gate/best.pt")
    ap.add_argument("--main-ckpt", default=f"{CK}/main_prof654.pth")
    ap.add_argument("--a2-ckpt", default=f"{CK}/a2_v7b_d128_alpha075_best_loss.pt")
    ap.add_argument("--seg-ckpt", default="outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt")  # i_1 cache 와 동일
    ap.add_argument("--backbone", default="tf_efficientnetv2_s")
    ap.add_argument("--fx", type=float, default=640.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs_multi/viz/gradcam_prof635"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mf = json.load(open(args.manifest)); track, policy, raw = mf["track"], mf["policy"], mf["raw_base"]

    bb = timm.create_model(args.backbone, pretrained=False, features_only=True, out_indices=[-1])
    sd = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False); own = bb.state_dict()
    for k,v in sd.items():
        k2=k.replace("backbone.","");
        if k2 in own and own[k2].shape==v.shape: own[k2]=v
    bb.load_state_dict(own); bb=bb.to(dev).eval(); [p.requires_grad_(False) for p in bb.parameters()]
    a2 = A2Model(img_backbone=args.backbone, d_token=256, n_heads=4, n_ch=7, d_out=128).to(dev).eval()
    sda = torch.load(args.a2_ckpt, map_location=dev, weights_only=False); sda = sda.get("model", sda) if isinstance(sda,dict) else sda
    a2.load_state_dict({k:v for k,v in sda.items() if k in a2.state_dict() and a2.state_dict()[k].shape==v.shape}, strict=False)
    [p.requires_grad_(False) for p in a2.parameters()]
    model = Prof635SideModel().to(dev); model.load_main_frozen(args.main_ckpt)
    if os.path.exists(args.model_ckpt):
        model.load_state_dict(torch.load(args.model_ckpt, map_location=dev, weights_only=False), strict=False)
        print(f"[model] f_2 loaded {args.model_ckpt}", flush=True)
    else:
        print(f"[model] f_2 ckpt 없음({args.model_ckpt}) — untrained 플러밍만", flush=True)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    pb = np.load(args.prebake, allow_pickle=True); ck=pb["curr_keys"].tolist(); pk=pb["prev_keys"].tolist()
    tr=pb["traj"]; eg=pb["ego_5d"]; tg=pb["target"]; dy=pb["d_yaws"]
    zg=np.load(args.inlane_gt, allow_pickle=True); zk={k:i for i,k in enumerate(zg["keys"].tolist())}
    gd=zg["inlane_dist"]; gc=zg["inlane_closing"]
    cand=[]
    for i,k in enumerate(ck):
        gi=zk.get(k)
        if gi is not None and np.isfinite(gd[gi]) and gd[gi]<20 and gc[gi]>1: cand.append((i,float(gd[gi]),float(gc[gi])))
    np.random.RandomState(0).shuffle(cand); pick=cand[:args.n]
    print(f"[lead-TTC] {len(cand)} → {len(pick)}", flush=True)
    rec=[]
    for n,(i,dd,cc) in enumerate(pick):
        rc=load_roi(raw,track,policy,*parse_key(ck[i])); rp=load_roi(raw,track,policy,*parse_key(pk[i]))
        if rc is None or rp is None: continue
        ic=norm(rc).unsqueeze(0).to(dev); ip=norm(rp).unsqueeze(0).to(dev); ic.requires_grad_(True)
        fc=bb(ic)[-1]; fc.retain_grad(); fp=bb(ip)[-1]
        tc=F.adaptive_avg_pool2d(fc,(1,40)).squeeze(2).transpose(1,2); tp=F.adaptive_avg_pool2d(fp,(1,40)).squeeze(2).transpose(1,2)
        dyt=torch.tensor([float(dy[i])],device=dev); fxt=torch.tensor([args.fx],device=dev)
        tpr=yaw_shift_tokens_batch(tp,dyt,fx=fxt)
        tcp=a2.encoder.proj(tc.transpose(1,2)).transpose(1,2); tpp=a2.encoder.proj(tpr.transpose(1,2)).transpose(1,2)
        e5=torch.from_numpy(eg[i]).unsqueeze(0).float().to(dev)
        ego_mot=torch.zeros(1,2,device=dev)  # build_i2_cache.py 와 동일 (ego_mot=0)
        fim=a2.fusion(tcp,tpp,ego_mot); i2=a2.fc_proj(fim); a2l=a2.head(i2)
        traj6=torch.from_numpy(build_traj6(tr[i],mode="speed")).unsqueeze(0).to(dev)
        ego6=torch.from_numpy(build_ego6(eg[i])).unsqueeze(0).to(dev)
        steer,fped=model({"traj6":traj6,"ego6":ego6,"i2_mem":i2,"a2_logits":a2l})
        g=float(model.last_gate.item()); base=float(model.last_base_pedal.item()); f2=float(model.last_f2_pedal.item())
        Lbrake=-fped[0]; model.zero_grad(set_to_none=True)
        if fc.grad is not None: fc.grad=None
        Lbrake.backward()
        A=fc[0].detach(); G=fc.grad[0].detach(); cam=torch.relu((G.mean((1,2))[:,None,None]*A).sum(0)).cpu().numpy()
        if cam.max()>0: cam/=cam.max()
        cam=np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((IMG_W,ROI_H),Image.BILINEAR))/255
        obj=1/(1+np.exp(-a2l[0,:,0].detach().cpu().numpy()))
        fig,ax=plt.subplots(2,1,figsize=(15,7),gridspec_kw=dict(height_ratios=[1.2,0.5]))
        ax[0].imshow(rc); ax[0].imshow(cam,cmap="jet",alpha=0.45)
        ax[0].set_title(f"{ck[i]} dist={dd:.1f}m clos={cc:+.1f} | gate g={g:.2f} base_pedal(f1)={base:+.2f} f2={f2:+.2f} final={float(fped):+.2f} steer={float(steer)*40:+.1f}° (brake attribution)",fontsize=10)
        ax[0].axis("off")
        for b in range(40):
            x0=b/40*IMG_W;
            if 16<=b<=25: ax[1].axvspan(x0,(b+1)/40*IMG_W,ymin=0,ymax=0.05,color="lime",alpha=.5)
            if obj[b]>0.5: ax[1].axvspan(x0,(b+1)/40*IMG_W,ymin=0.05,ymax=0.05+0.4*obj[b],color="red",alpha=.4)
        ax[1].set_xlim(0,IMG_W); ax[1].set_title("a_2 obj (red) | in-lane 16-25 (green)",fontsize=9); ax[1].axis("off")
        fn=f"{n:02d}_{ck[i].replace('__','_')}_d{dd:.0f}.png"; plt.tight_layout(); plt.savefig(args.out/fn,dpi=100); plt.close()
        rec.append((ck[i],dd,cc,g,base,f2,float(fped),fn))
        print(f"  [{n+1}] {fn} g={g:.2f} base={base:+.2f} f2={f2:+.2f} final={float(fped):+.2f}",flush=True)
    html=args.out/"index.html"
    with open(html,"w") as f:
        f.write("<!doctype html><meta charset=utf-8><style>body{background:#1c1c1c;color:#ddd;font-family:sans-serif}.c{margin:10px;background:#2a2a2a;padding:6px}.c img{width:100%}</style>")
        f.write(f"<h2>Prof635 (#654+f_2 gated) brake grad-cam — lead-TTC</h2>")
        for k,dd,cc,g,base,f2,fp,fn in rec:
            f.write(f"<div class=c><div style='font-size:12px;color:#aaa'>{k} d={dd:.1f} c={cc:+.1f} | g={g:.2f} base={base:+.2f} f2={f2:+.2f} final={fp:+.2f}</div><img src='{fn}'></div>")
    print(f"[done] {html} ({len(rec)})",flush=True)


if __name__ == "__main__":
    main()
