from typing import List, Optional

import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, to_hex, to_rgba

GREEN = "#70B375"
ORANGE = "#B37570"
PURPLE = "#7570B3"


def get_colors(num_colors: int, alpha: Optional[float] = 1) -> List[List[float]]:
    """Get a list of colorblind samples_colors,
    :param num_colors: Number of samples_colors.
    :param alpha: The transparency
    :return: List of samples_colors. Each color is a list of [r, g, b, alpha].
    """
    palettes = ["colorblind", "ch:start=.2,rot=-.3"]
    cs = sns.color_palette(palettes[0], n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs


def make_colormap_to_white(color="tab:orange"):
    color_rgb = np.array(to_rgba(color))
    lower = np.ones((int(256 / 4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, color_rgb[i], lower.shape[0])
    cmap = np.vstack(lower)
    return ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])


def get_alpha_colormap(hex_color, level):
    rbga = to_rgba(hex_color)
    return (to_hex((rbga[0], rbga[1], rbga[2], l), True) for l in level)
