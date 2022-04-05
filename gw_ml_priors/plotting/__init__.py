import os

import matplotlib.pyplot as plt


def set_style():
    fname = os.path.join(os.path.basename(__file__), "plotting.mplstyle")
    plt.style.use(fname)
