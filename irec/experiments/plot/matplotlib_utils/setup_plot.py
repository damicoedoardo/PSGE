#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import os

# todo change path to relative path
# STYLESHEET_PATH = "../styles/"
STYLESHEET_PATH = (
    os.environ["HOME"] + "/GitProjects/InductiveRec/irec/experiments/"
    "plot/matplotlib_utils/styles/"
)


def setup_plot(w_pts=234, fig_ratio=0.69, font_size=8, dpi=None, style_sheet="base"):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import os

    # Convert pt to inches
    inches_per_pt = 1.0 / 72.27
    # first set the mpl defaults, then␣,!load our style
    mpl.rcdefaults()
    plt.style.use(STYLESHEET_PATH + style_sheet + ".mplstyle")
    plt.rc("font", family="Computer Modern")
    # Sometime need to quickly adjust font size! so include it as an option...
    if font_size is not None:
        mpl.rcParams.update({"font.size": font_size})
    # convert pts to matplotlib! dimensions
    w = w_pts * inches_per_pt
    h = w * fig_ratio
    # dpi only matters for png
    dpi = 120 if dpi is None else dpi
    mpl.rcParams.update({"figure.figsize": (w, h)})
    mpl.rcParams.update({"figure.dpi": dpi})

    # tex
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{cmbright}",
        }
    )
    return
