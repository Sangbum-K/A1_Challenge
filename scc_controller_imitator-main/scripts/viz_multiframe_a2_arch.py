"""multi-frame a_2 (perception) 모듈 설계 다이어그램. 라벨 영문(한글 폰트 깨짐 방지)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PRI, ACC, DARK, GRAY, WARN = "#005195", "#01A0E9", "#002060", "#888888", "#DC291E"
fig, ax = plt.subplots(figsize=(15, 8.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, txt, fc="white", ec=PRI, fs=9, tc=DARK, lw=1.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                 fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, c=DARK, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=12, color=c, lw=lw))


ax.text(7.5, 8.75, "multi-frame a_2 (perception) — relative-velocity / closing estimation",
        ha="center", fontsize=14, fontweight="bold", color=DARK)

# ---- left: N+1 frames (uniform 5-tick) ----
frames = [("curr  t",      7.7, ACC),
          ("prev  t-5",    6.3, GRAY),
          ("prev  t-10",   4.9, GRAY),
          ("prev  t-15",   3.5, GRAY),
          ("prev  t-20",   2.1, GRAY)]
ax.text(1.15, 8.35, "frames (uniform 5-tick)", ha="center", fontsize=9, color=DARK, fontweight="bold")
for name, yy, c in frames:
    box(0.3, yy, 1.7, 0.75, name, ec=c, tc=DARK, fs=8.5)
    # seg backbone -> i_1 tokens
    box(2.45, yy, 1.5, 0.75, "seg bb\n-> i_1 (40,d)", ec=c, fs=7.5)
    arrow(2.0, yy + 0.37, 2.45, yy + 0.37, c=c)
    arrow(3.95, yy + 0.37, 5.1, yy + 0.37, c=c)
ax.annotate("", xy=(0.15, 2.1), xytext=(0.15, 8.45),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1))
ax.text(0.0, 5.3, "time", rotation=90, va="center", fontsize=8, color=DARK)

# ---- KV stack ----
box(5.1, 1.9, 2.6, 6.05, "", ec=PRI, lw=1.8)
ax.text(6.4, 8.15, "Temporal stack = KV", ha="center", fontsize=9.5, color=PRI, fontweight="bold")
for name, yy, c in frames:
    if "curr" in name:
        box(5.3, yy, 2.2, 0.55, "curr tokens (= Q)", ec=ACC, fs=7.8, tc=PRI, bold=True)
    else:
        box(5.3, yy, 2.2, 0.55, "tokens (40,d)", ec=c, fs=7.5)
ax.text(6.4, 1.5, "+ lat_pos  + temporal_emb(n)\n+ ego_emb(dYaw,yawrate,vx)\n[no hard yaw-shift]",
        ha="center", fontsize=7.6, color=WARN)

# ---- attention blocks (vertical sequence) ----
box(8.3, 6.3, 2.7, 1.35, "lateral self-attn\n(per frame: perceive\nobject WIDTH / span)", ec=PRI, fs=8.5)
box(8.3, 3.95, 2.7, 1.7, "temporal cross-attn  x L\nQ(curr) x KV(stack)\nmatch object across frames\n-> d(width,content)/dt", ec=PRI, fs=8.2, bold=True)
box(8.3, 2.2, 2.7, 1.05, "fc_proj -> i_2 mem (40,128)", ec=PRI, fs=8.5)
arrow(7.7, 7.0, 8.3, 6.97, c=PRI)            # KV -> self-attn
arrow(7.7, 7.95, 9.65, 5.65, c=ACC)          # Q(curr) -> temporal cross-attn
ax.text(8.05, 7.5, "Q", color=ACC, fontsize=9, fontweight="bold")
arrow(9.65, 6.3, 9.65, 5.65, c=PRI)          # self -> temporal
arrow(7.7, 4.8, 8.3, 4.8, c=PRI)             # KV -> temporal
arrow(9.65, 3.95, 9.65, 3.25, c=PRI)         # temporal -> fc_proj

# ---- head + outputs ----
box(11.3, 3.4, 3.4, 1.7, "a_2 head (Linear)\n[obj, vL_abs, vY_abs,\n vL_rel, vY_rel, prox, dist]\n(per column, 40 x 7)", ec=PRI, fs=8.3, bold=True)
arrow(11.0, 2.72, 11.3, 4.0, c=PRI)
box(11.3, 1.4, 3.4, 1.3, "i_2 mem -> f_2 (side) gate+brake\nvL_rel -> closing-based brake", ec=ACC, fs=8, tc=PRI)
arrow(11.0, 2.72, 11.3, 2.05, c=ACC)

# ---- key insight ----
ax.text(11.3, 6.55, "KEY: object width (column span)\ngrows as it APPROACHES\n-> d(width)/dt = closing\n(lateral strip retains this;\n content magnitude = height cue)",
        ha="left", va="center", fontsize=8.5, color=WARN,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff4f4", ec=WARN, lw=1.2))

# goal note
ax.text(7.5, 0.5, "Goal: fix a_2 vL_rel (corr 0.28 -> higher) — 2-frame finite-diff is too noisy; "
        "N-frame temporal attention estimates velocity & acceleration of object width/content.",
        ha="center", fontsize=8.5, color=DARK)

plt.tight_layout()
out = "outputs_multi/viz/multiframe_a2_arch.png"
import os; os.makedirs("outputs_multi/viz", exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
print(f"[done] {out}")
