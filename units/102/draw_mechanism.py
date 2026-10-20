from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tutorial-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/tutorial-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.axis('off')
labels = ['Unique docs\nD_unique', 'Repeated draws\nD_processed', 'Model width\nN parameters', 'Loss + time\nMeasured']
for i, label in enumerate(labels):
    ax.text(0.12 + i * 0.25, 0.46, label, ha='center', va='center', fontsize=14, bbox={'boxstyle': 'round,pad=.45', 'facecolor': ['#e6f1ff', '#fff0d8', '#e6f1ff', '#e3f6e9'][i]})
    if i < 3:
        ax.annotate('', xy=(0.282 + i * 0.25, 0.46), xytext=(0.217 + i * 0.25, 0.46), arrowprops={'arrowstyle': '->', 'lw': 1.5})
ax.set_title('Budget audit: equal steps ≠ equal compute', fontsize=15, pad=8)
fig.tight_layout()
fig.savefig(ROOT / 'figures/mechanism.png', dpi=180)
plt.close(fig)
