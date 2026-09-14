"""IyModel architecture diagram — paper-style with tensor shapes.

Vertical layout: inputs (top) → image+text branches → concat → transformer → outputs (bottom).
Outputs: outputs_multi/viz_resynth/arch/iy_architecture.png (+ .svg).
"""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

# ───────── style ─────────
C_IMG   = "#D6EAF8"   # image stream
C_TEXT  = "#FAE5D3"   # text stream
C_CORE  = "#E5E7E9"   # shared / fusion / transformer
C_FROZ  = "#F2F4F4"   # frozen / cache
C_OUT   = "#FADBD8"   # output
C_GATE  = "#F5CBA7"   # v3 new (i_g)
C_LOSS  = "#FDEDEC"   # supervision loss
NAVY    = "#1a1a2e"
RED     = "#C0392B"

def box(ax, xy, w, h, label, shape="", color=C_CORE, edge=NAVY, lw=1.3,
        ls="-", fs_label=11, fs_shape=9, badge=None, badge_color="#28B463"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=edge, facecolor=color, linestyle=ls)
    ax.add_patch(p)
    if shape:
        ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center",
                fontsize=fs_label, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + h * 0.28, shape, ha="center", va="center",
                fontsize=fs_shape, color="#34495E", family="monospace")
    else:
        ax.text(x + w / 2, y + h * 0.5, label, ha="center", va="center",
                fontsize=fs_label, fontweight="bold", color=NAVY)
    if badge:
        ax.text(x + w - 0.05, y + h - 0.07, badge, ha="right", va="top",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=badge_color, edgecolor="none"))
    return (x + w / 2, y, x + w / 2, y + h)   # bottom-mid, top-mid

def arrow(ax, p_from, p_to, label="", ls="-", color=NAVY, lw=1.2, label_off=(0.05, 0)):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=14,
                         linewidth=lw, color=color, linestyle=ls,
                         connectionstyle="arc3,rad=0")
    ax.add_patch(a)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx + label_off[0], my + label_off[1], label,
                ha="left", va="center", fontsize=8, color="#566573",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

# ───────── layout ─────────
fig, ax = plt.subplots(figsize=(14, 17))
ax.set_xlim(0, 14); ax.set_ylim(0, 22.5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_facecolor("white")

# title
ax.text(7, 22.0, "IyModel — End-to-End SCC Controller (v3)",
        ha="center", va="center", fontsize=17, fontweight="bold", color=NAVY)
ax.text(7, 21.5, "i_1 cache + i_y token-level Δyaw shift + i_g obj-gate + Professor f_1 transformer",
        ha="center", va="center", fontsize=10, color="#566573", style="italic")

# ───── INPUTS (top row) ─────
y_in = 19.7
w_in = 2.2; h_in = 0.95
inp_specs = [
    ("image_curr", "(B, 3, 250, 1280)", C_IMG,  0.6),
    ("image_prev", "(B, 3, 250, 1280)", C_IMG,  3.2),
    ("d_yaw",       "(B,)  rad",         C_TEXT, 5.8),
    ("ego_5d",      "(B, 5)\n[vx,vy,ax,ay,wz]", C_TEXT, 8.4),
    ("traj_40",     "(B, 40, 3)\n[x,y,speed]",  C_TEXT, 11.0),
]
inp_centers = {}
for name, shape, color, x in inp_specs:
    btm, _, top, _ = box(ax, (x, y_in), w_in, h_in, name, shape, color=color)
    inp_centers[name] = (x + w_in / 2, y_in)   # bottom

# input bracket label
ax.text(7, 20.95, "Inputs", ha="center", va="center", fontsize=11,
        fontweight="bold", color="#566573")

# ───── IMAGE BRANCH (left column) ─────
xL = 1.4; wL = 5.0

# i_1: LateralStripEncoder (cached, shared)
y1 = 17.5
box(ax, (xL, y1), wL, 1.2,
    "i_1 · LateralStripEncoder",
    "EfficientNetV2-S backbone (seg-pretrained, frozen) → AdaptiveAvgPool2d((1,40)) → squeeze → transpose",
    color=C_FROZ, ls=(0, (4, 2)), badge="❄ cached", badge_color="#5D6D7E", fs_shape=8)

# i_y: token-level shift
y2 = 15.7
box(ax, (xL, y2), wL, 1.1,
    "i_y · Token-level Δyaw shift",
    "dx = round(fx · d_yaw / 32) ;  np.roll(fi_tok_prev, dx, axis=lateral)",
    color=C_IMG, fs_shape=8)

# img_proj
y3 = 14.0
box(ax, (xL, y3), wL, 1.0,
    "img_proj · Conv1d(256 → 128, k=1)  shared",
    "fi_tok_curr → tok_c    ;    fi_tok_prev_y → tok_p",
    color=C_IMG, fs_shape=8)

# i_2: ImageFusionBlock - 3 stages
y4 = 12.4
box(ax, (xL, y4), wL, 1.1,
    "i_2 · Stage 0 — per-frame SelfAttn (shared, 1L)",
    "fc' = SA(tok_c) ;  fp' = SA(tok_p)",
    color=C_CORE, fs_shape=8)

y5 = 10.8
box(ax, (xL, y5), wL, 1.2,
    "i_2 · Stage 1 — CrossAttn",
    "Q = fc'   K/V = concat([ego_motion_tok, fp'])  ;  ego_motion_tok = Linear(2→128)([vx, wz])",
    color=C_CORE, fs_shape=8)

y6 = 9.2
box(ax, (xL, y6), wL, 1.1,
    "i_2 · Stage 2 — Post-fusion SelfAttn (40 tokens)",
    "→ fi_mem",
    color=C_CORE, fs_shape=8)

# i_g: gate
y7 = 7.5
box(ax, (xL, y7), wL, 1.2,
    "i_g · Image Gate",
    "obj_logit = Linear(128→1)(fi_mem) ;  gate = σ(obj_logit) ;  fi_mem_gated = fi_mem · gate",
    color=C_GATE, badge="v3", badge_color=RED, fs_shape=8)

# ───── TEXT BRANCH (right column) ─────
xR = 8.0; wR = 5.0

ty1 = 17.5
box(ax, (xR, ty1), wR, 1.0,
    "traj_embed · Linear(3 → 128)",
    "traj_40 → traj_tok",
    color=C_TEXT, fs_shape=8)

ty2 = 15.9
box(ax, (xR, ty2), wR, 1.0,
    "state_embed · Linear(5 → 128)",
    "ego_5d → state_tok",
    color=C_TEXT, fs_shape=8)

ty3 = 14.3
box(ax, (xR, ty3), wR, 1.1,
    "Add@t=0  +  PositionalEncoding (max_len=41)",
    "traj_tok[:, 0, :] += state_tok ;  tokens_text",
    color=C_TEXT, fs_shape=8)

# ───── MERGE & DECODE ─────
yC = 6.0
box(ax, (3.0, yC), 8.0, 1.0,
    "Concat along token axis  +  image_pos_emb",
    "concat([tokens_text, fi_mem_gated + image_pos_emb])  →  tokens",
    color=C_CORE, fs_shape=8)

yT = 4.2
box(ax, (3.0, yT), 8.0, 1.4,
    "f_1 · TransformerEncoder (6 layers, d_model=128, nhead=8, FFN=512)",
    "Professor #654 warm-start  →  encoded",
    color=C_CORE, lw=2.0, badge="Professor", badge_color="#5D6D7E", fs_shape=8)

yF = 2.8
box(ax, (3.0, yF), 8.0, 0.9,
    "fc_out · Linear(128 → 2)  on first 40 tokens",
    "→ pred (B, 40, 2)  =  [steer_deg, ax_mps²]",
    color=C_CORE, fs_shape=8)

# ───── OUTPUTS ─────
yO = 1.2
box(ax, (1.5, yO), 2.8, 0.9, "steer", "(B, 40)  deg",
    color=C_OUT, fs_shape=8)
box(ax, (5.6, yO), 2.8, 0.9, "ax",    "(B, 40)  m/s²",
    color=C_OUT, fs_shape=8)
box(ax, (9.7, yO), 2.8, 0.9, "obj_logit", "(B, 40)  (i_g)",
    color=C_OUT, fs_shape=8)

# ───── SUPERVISION (right margin) ─────
xSup = 12.5
box(ax, (xSup, 3.5), 1.3, 0.7, "L1\nsteer/10", "", color=C_LOSS, edge=RED, fs_label=8)
box(ax, (xSup, 2.3), 1.3, 0.7, "L1\nax/10",    "", color=C_LOSS, edge=RED, fs_label=8)
box(ax, (xSup, 1.0), 1.3, 0.7, "BCE\n× 0.5",   "", color=C_LOSS, edge=RED, fs_label=8)

# ────────────────────────────────────────────────────────────────
# ARROWS
# ────────────────────────────────────────────────────────────────
# inputs → image branch
arrow(ax, inp_centers["image_curr"], (xL + 1.5, 17.5 + 1.2), "(B,3,250,1280)")
arrow(ax, inp_centers["image_prev"], (xL + 3.5, 17.5 + 1.2), "(B,3,250,1280)")
# d_yaw → i_y
arrow(ax, inp_centers["d_yaw"], (xL + wL - 0.4, 15.7 + 1.1), "scalar")

# inputs → text branch
arrow(ax, inp_centers["traj_40"], (xR + wR - 1.0, ty1 + 1.0), "(B,40,3)")
arrow(ax, inp_centers["ego_5d"],  (xR + 1.0, ty2 + 1.0), "(B,5)")
# ego[:2] (vx, wz) → i_2 Stage 1 (ego_motion_token)
arrow(ax, (inp_centers["ego_5d"][0]-0.5, inp_centers["ego_5d"][1]),
      (xL + wL - 0.3, y5 + 0.6),
      "vx, wz", ls="-", color="#5D6D7E", lw=1.0, label_off=(0.05, 0.3))

# image branch chain
arrow(ax, (xL + wL / 2, 17.5), (xL + wL / 2, 15.7 + 1.1),
      "fi_tok_curr/prev\n(B, 40, 256)", label_off=(0.15, 0))
arrow(ax, (xL + wL / 2, 15.7), (xL + wL / 2, 14.0 + 1.0),
      "(B, 40, 256)")
arrow(ax, (xL + wL / 2, 14.0), (xL + wL / 2, 12.4 + 1.1),
      "(B, 40, 128)")
arrow(ax, (xL + wL / 2, 12.4), (xL + wL / 2, 10.8 + 1.2),
      "fc', fp' (B, 40, 128)")
arrow(ax, (xL + wL / 2, 10.8), (xL + wL / 2, 9.2 + 1.1),
      "(B, 40, 128)")
arrow(ax, (xL + wL / 2, 9.2), (xL + wL / 2, 7.5 + 1.2),
      "fi_mem (B, 40, 128)")

# i_g → concat + obj_logit out
arrow(ax, (xL + wL / 2 - 1.0, 7.5), (4.0, yC + 1.0),
      "fi_mem_gated\n(B, 40, 128)")
arrow(ax, (xL + wL / 2 + 1.5, 7.5), (10.7, yO + 0.9),
      "obj_logit", color="#5D6D7E", ls=(0, (3, 2)))

# text branch chain
arrow(ax, (xR + wR / 2, ty1), (xR + wR / 2, ty2 + 1.0),
      "traj_tok (B, 40, 128)")
arrow(ax, (xR + wR / 2 + 1.0, ty2), (xR + wR / 2 + 1.0, ty3 + 1.1),
      "state_tok (B, 1, 128)")
arrow(ax, (xR + wR / 2, ty3), (10.0, yC + 1.0),
      "tokens_text\n(B, 40, 128)")

# concat → transformer → fc_out → outputs
arrow(ax, (7, yC), (7, yT + 1.4), "tokens (B, 80, 128)")
arrow(ax, (7, yT), (7, yF + 0.9), "encoded (B, 80, 128)")
arrow(ax, (7 - 2, yF), (1.5 + 1.4, yO + 0.9), "")
arrow(ax, (7, yF),     (5.6 + 1.4, yO + 0.9), "(B, 40, 2)")

# supervision (red dashed) — target dots
def _red_dash(ax, p1, p2, txt):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.0, color=RED, linestyle=(0, (4, 2)))
    ax.add_patch(a)
    if txt:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.08, txt, ha="center", va="bottom", fontsize=7, color=RED,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8))

_red_dash(ax, (1.5 + 1.4, yO + 0.45),  (xSup, 3.5 + 0.35), "vs target_steer")
_red_dash(ax, (5.6 + 1.4, yO + 0.45),  (xSup, 2.3 + 0.35), "vs target_ax")
_red_dash(ax, (9.7 + 1.4, yO + 0.45),  (xSup, 1.0 + 0.35), "vs obj_target")

# group labels (image / text branch brackets)
ax.text(xL + wL / 2, 18.7, "── Image branch ──", ha="center", va="center",
        fontsize=10, color="#566573", fontweight="bold")
ax.text(xR + wR / 2, 18.7, "── Text branch ──", ha="center", va="center",
        fontsize=10, color="#566573", fontweight="bold")
ax.text(7.0, yC + 1.5, "── Merge & decode ──", ha="center", va="center",
        fontsize=10, color="#566573", fontweight="bold")

# legend
leg_y = 0.2
ax.text(0.3, leg_y, "stream:", fontsize=9, va="center", color=NAVY)
for lx, label, color in [(1.5, "image", C_IMG), (3.0, "text", C_TEXT),
                          (4.4, "shared", C_CORE), (5.8, "frozen", C_FROZ),
                          (7.2, "v3 new", C_GATE), (8.6, "loss", C_LOSS),
                          (9.9, "output", C_OUT)]:
    ax.add_patch(FancyBboxPatch((lx, leg_y - 0.18), 0.4, 0.36,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=color, edgecolor=NAVY, linewidth=0.6))
    ax.text(lx + 0.45, leg_y, label, fontsize=8, va="center", color=NAVY)

# footer info
ax.text(13.7, leg_y, "fx=640 px,  cx=640,  bin_w=32 px,  T=40 (4s)\nparams≈1.63M (excl. frozen backbone)",
        fontsize=7, ha="right", va="center", color="#566573", family="monospace")

# save
out_dir = Path("outputs_multi/viz_resynth/arch")
out_dir.mkdir(parents=True, exist_ok=True)
png = out_dir / "iy_architecture.png"
svg = out_dir / "iy_architecture.svg"
plt.tight_layout()
plt.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(svg, bbox_inches="tight", facecolor="white")
plt.close()
print(f"saved → {png}")
print(f"saved → {svg}")
