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
for i, (top, bottom) in enumerate(zip(['BOS', 'the ', 'red ', 'cat'], ['the ', 'red ', 'cat', 'EOS'])):
    x = 0.12 + i * 0.25
    ax.text(x, 0.76, top, ha='center', fontsize=17, bbox={'boxstyle': 'round,pad=.4', 'facecolor': '#dceefd'})
    ax.annotate('', xy=(x, 0.37), xytext=(x, 0.63), arrowprops={'arrowstyle': '->', 'lw': 1.7})
    ax.text(x, 0.19, bottom, ha='center', fontsize=17, bbox={'boxstyle': 'round,pad=.4', 'facecolor': '#def5e7'})
ax.set_title('Input at t → prediction of token t+1', fontsize=15, pad=15)
fig.tight_layout()
fig.savefig(ROOT / 'figures/mechanism.png', dpi=180)
plt.close(fig)
