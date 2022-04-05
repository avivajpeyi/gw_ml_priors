import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deep_gw_pe_followup import get_mpl_style

sns.set_theme(style="ticks")
plt.style.use(get_mpl_style())

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]

CMAP = "hot"


def plot_probs(x, y, p, xlabel, ylabel, plabel=None, fname=None):
    plt.close("all")

    fig, axes = plt.subplots(2, 1, figsize=(4, 8))
    ax = axes[0]

    try:
        if isinstance(p, pd.Series):
            z = p.values
        else:
            z = p.copy()
        z[z == -np.inf] = np.nan

        ax.tricontour(x, y, z, 15, linewidths=0.5, colors="k")

        cmap = ax.tricontourf(
            x,
            y,
            z,
            15,
            vmin=np.nanmin(z),
            vmax=np.nanmax(z),
            # norm=plt.Normalize(vmax=abs(p).max(), vmin=-abs(p).max()),
            cmap=CMAP,
        )
    except Exception:
        cmap = ax.scatter(x, y, c=p, cmap=CMAP)

    if plabel:
        fig.colorbar(cmap, label=plabel, ax=ax)

    plot_heatmap(x, y, z, axes[1], plabel=plabel)

    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_aspect(1.0 / ax.get_data_ratio())

    if fname:
        fig.tight_layout()
        fig.savefig(fname)
    else:
        return fig, axes


def plot_heatmap(x, y, p, ax, plabel=None, cmap=CMAP):
    if isinstance(p, pd.Series):
        z = p.values
        x = x.values
        y = y.values
    else:
        z = p.copy()
    z[z == -np.inf] = np.nan

    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Z = z.reshape(len(x), len(y))

    cmap = ax.pcolor(
        X, Y, Z, cmap=cmap, vmin=np.nanmin(z), vmax=np.nanmax(z), zorder=-100
    )

    if plabel:
        fig = ax.get_figure()
        fig.colorbar(cmap, ax=ax, label=plabel)
