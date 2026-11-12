import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frontier_database import PUBLISHED_FRONTIER_MODELS

fig_dir = "/Users/nguyenvokhang/Downloads/Create-LLM/LLM-model/figures"
artifact_dir = "/Users/nguyenvokhang/.gemini/antigravity/brain/e5f06cbf-d442-4fbf-8857-700a2cc4cdc6"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

models_d = PUBLISHED_FRONTIER_MODELS["dense"]

names = [m["name"].replace("Dense ", "").replace(" (Alba Innovation)", "").replace(" (Alba)", "") for m in models_d]
depth_ratios = [m["depth_ratio"] * 100 for m in models_d]
colors = ['#f97316', '#16a34a', '#3b82f6', '#0284c7', '#6366f1', '#4f46e5', '#64748b', '#0d9488', '#059669']

y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, depth_ratios, color=colors, height=0.65, edgecolor='#000000', linewidth=0.8)

ax1.axvline(37.0, color='#ef4444', linestyle=':', linewidth=1.5, label='Classical Residual Baseline (~37%)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlabel('Effective Depth Utilization Ratio: (D_eff / L) [%]', fontsize=10, fontweight='bold')
ax1.set_title('(a) Effective Depth Utilization vs. Frontier Models\n(Gahtan et al., ACML 2026 Reference Suite)', fontsize=10.5, fontweight='bold')
ax1.set_xlim(0, 65)
ax1.grid(True, axis='x')

for bar in bars:
    w = bar.get_width()
    ax1.text(w + 1.0, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=8.5, fontweight='bold')

ax1.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8)

all_models = []
for m in PUBLISHED_FRONTIER_MODELS["dense"]:
    all_models.append((m["active_params_m"], m["val_loss"], m["name"], "dense"))
for m in PUBLISHED_FRONTIER_MODELS["moe"]:
    all_models.append((m["active_params_m"], m["val_loss"], m["name"], "moe"))

for active_m, val_l, name, m_type in all_models:
    if "Alba" in name:
        if "OS-MoE" in name:
            ax2.scatter(active_m, val_l, color='#7c3aed', marker='*', s=240, zorder=6)
            ax2.annotate(name, (active_m, val_l), textcoords="offset points", xytext=(-25, 12), fontsize=8, fontweight='bold', color='#6b21a8')
        else:
            ax2.scatter(active_m, val_l, color='#16a34a', marker='o', s=130, zorder=5)
            ax2.annotate(name, (active_m, val_l), textcoords="offset points", xytext=(8, -10), fontsize=8, fontweight='bold', color='#15803d')
    elif m_type == "moe":
        ax2.scatter(active_m, val_l, color='#ec4899', marker='D', s=80, zorder=4)
        ax2.annotate(name, (active_m, val_l), textcoords="offset points", xytext=(8, -5), fontsize=7.5, color='#be185d')
    else:
        ax2.scatter(active_m, val_l, color='#3b82f6', marker='s', s=70, zorder=4)
        ax2.annotate(name, (active_m, val_l), textcoords="offset points", xytext=(8, 4), fontsize=7.5, color='#1d4ed8')

ax2.set_xscale('log')
ax2.set_title('(b) Multi-Scale Pareto Frontier (51M to 16B)\n(DeepSeekMoE & Moonlight 2026 Style)', fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Active Parameters per Token (Millions, Log-Scale)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Validation Loss / Perplexity Equivalent', fontsize=10, fontweight='bold')
ax2.set_ylim(0.12, 0.30)
ax2.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "frontier_master_comparison.png"), dpi=300)
plt.savefig(os.path.join(artifact_dir, "frontier_master_comparison.png"), dpi=300)
plt.close()
print("Frontier master comparison figure generated successfully.")
