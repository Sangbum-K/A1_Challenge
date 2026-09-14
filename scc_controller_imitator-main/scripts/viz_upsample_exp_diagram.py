"""업샘플 검증 실험 다이어그램 (3 config). 라벨 영문(폰트 깨짐 방지)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PRI, ACC, DARK, GRAY, WARN, GRN = "#005195", "#01A0E9", "#002060", "#888888", "#DC291E", "#1a8a3a"
fig, ax = plt.subplots(figsize=(15, 8.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")

def box(x,y,w,h,txt,fc="white",ec=PRI,fs=9,tc=DARK,lw=1.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03",fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,c=DARK,lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color=c,lw=lw))

ax.text(7.5,8.7,"Upsample verification — does 1-layer U-Net help 2-frame vL_rel?",ha="center",fontsize=14,fontweight="bold",color=DARK)

# encoder
box(0.3,6.6,1.7,1.0,"image\n(250x1280)\ncurr & prev20",ec=ACC,fs=8)
box(2.3,6.6,2.0,1.0,"seg encoder\n(EffNetV2-S)\nfrozen",ec=PRI,fs=8.5)
arrow(2.0,7.1,2.3,7.1,c=DARK)
# pyramid
box(4.55,7.3,1.7,0.55,"f5  stride32  256ch  (~40 wide)",ec=GRAY,fs=7.5)
box(4.55,6.5,1.7,0.55,"f4  stride16  (~80 wide)",ec=GRAY,fs=7.5)
arrow(4.3,7.1,4.55,7.55,c=GRAY); arrow(4.3,7.0,4.55,6.77,c=GRAY)
# up4 (1-layer U-Net)
box(6.6,6.55,2.2,1.25,"up4 (1-layer U-Net)\nConvT(f5) + skip(f4)\n-> d4  stride16  128ch\n(~80 wide)",ec=GRN,fs=8,bold=True)
arrow(6.25,7.55,6.6,7.4,c=GRAY)       # f5 -> up4
arrow(6.25,6.77,6.6,6.95,c=GRN)       # f4 skip -> up4

# ---- 3 config pooling branches ----
# A: f5 -> pool40
box(9.2,7.55,2.3,0.62,"A: f5 -> pool(1,40)\n= current i_1 (40 tok)",ec=GRAY,fs=8,tc=DARK)
arrow(6.25,7.62,9.2,7.86,c=GRAY)
# B: d4 -> pool40
box(9.2,6.45,2.3,0.62,"B: d4 -> pool(1,40)\ndecode only, same res",ec=PRI,fs=8,tc=PRI)
arrow(8.8,7.0,9.2,6.76,c=PRI)
# C: d4 -> pool80
box(9.2,5.35,2.3,0.62,"C: d4 -> pool(1,80)\ndecode + 2x res (proposal)",ec=GRN,fs=8,tc=GRN,bold=True)
arrow(8.8,7.0,9.2,5.66,c=GRN)

# TinyVrel
box(12.0,5.8,2.7,1.9,"TinyVrel (same recipe)\nproj -> cross-attn\n(Q=curr, KV=prev20)\n-> mean -> closing scalar\ntrain: MSE(GT closing)",ec=PRI,fs=8.2,bold=True)
for yy in (7.86,6.76,5.66): arrow(11.5,yy,12.0,6.75,c=GRAY,lw=1)

# metric / decompose
box(2.5,3.3,10.0,1.5,"metric: val corr(pred closing, GT closing)   |   "
    "A->B = decode effect,  B->C = resolution effect,  A->C = full proposal gain",ec=DARK,fs=10,tc=DARK)
ax.text(7.5,2.55,"Hypothesis: far object width ~ 1 column (40-bin floor) -> does resolution (C) raise vL_rel?",ha="center",fontsize=9.5,color=WARN)
ax.text(7.5,1.9,"But corr-vs-distance: far NOT worst (-0.31 vs near -0.13) -> vL_rel signal may be content, not width.",ha="center",fontsize=9,color=DARK)
ax.text(7.5,1.35,"If effect negligible -> upsample redundant; bottleneck elsewhere (2-frame diff noise) -> go multi-frame.",ha="center",fontsize=9,color=DARK)
ax.text(7.5,0.7,"2-frame (curr, prev20) fixed. seg encoder/up4 frozen; only TinyVrel trained from scratch, identical.",ha="center",fontsize=8.5,color=GRAY)

import os; os.makedirs("outputs_multi/viz",exist_ok=True)
plt.tight_layout(); plt.savefig("outputs_multi/viz/upsample_exp_diagram.png",dpi=120,bbox_inches="tight"); plt.close()
print("[done] outputs_multi/viz/upsample_exp_diagram.png")
