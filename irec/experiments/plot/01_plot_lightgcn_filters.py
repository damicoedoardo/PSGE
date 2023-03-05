import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot
import os
from irec.experiments.plot.matplotlib_utils.setup_plot import setup_plot


def lightgcn_filter_function(x, k):
    y = (1 / (k + 1)) * ((1 - (1 - x) ** (k + 1)) / (x))
    return y

def inverse_lightgcn_filter_function(x, k):
    y=(1 / (k + 1)) * ((1 - x ** (k + 1)) / (1-x))
    return y


if __name__ == "__main__":
    setup_plot(234, fig_ratio=0.5, style_sheet="base")
    #xvals = np.linspace(0, 2, 1000)[1:]
    xvals=np.linspace(-1, 1, 1000)[1:-1]
    x = []
    y = []
    conv_depth = []

    for k in [1, 2, 3, 4]:
        x.extend(xvals)
        y.extend(inverse_lightgcn_filter_function(xvals, k))
        conv_depth.extend(np.repeat(k, len(xvals)))

    data_df = pd.DataFrame(
        zip(x, y, conv_depth), columns=["Lambda", "Spectral Coefficient", "k"]
    )

    # sizes = {1: 1, 2: 1.3, 3: 1.6, 4: 1.9, 5: 2.2, 6: 2.5}
    sizes={1: 1, 2: 1.4, 3: 1.8, 4: 2.2}

    # palette = sns.color_palette("Blues", 5)[1:]
    palette = sns.color_palette("viridis", 4)[::-1]
    # palette = sns.cubehelix_palette(start=2, rot=-.8)
    odd_color = "red"
    even_color = "teal"
    colors = {
        1: odd_color,
        2: even_color,
        3: odd_color,
        4: even_color,
        5: odd_color,
        6: even_color,
    }
    g = sns.lineplot(
        data=data_df,
        x="Lambda",
        y="Spectral Coefficient",
        hue="k",
        size="k",
        sizes=sizes,
        # style="k",
        palette=palette,
    )
    g.set(ylim=(0, 1))
    #g.set(xlim=(0, 2))
    g.set(xlim=(-1, 1))
    # g.xaxis.grid(False)

    # set legend to be horizontal
    # legend = plt.legend(title="$k$", ncol=1, columnspacing=0.15)
    legend = plt.legend(labels=["$k=1$", "$k=2$", "$k=3$", "$k=4$"])
    legend.get_frame().set_linewidth(0.5)

    # top
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
    #           fancybox=False, ncol=6, columnspacing=0.4)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
    #            fancybox=False, ncol=6, columnspacing=0.4)
    # setting labels
    plt.ylabel("$\\hat{g}(\\lambda$)")
    plt.xlabel("$\\lambda$")

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(
        "{}/egcn_plot/{}.pdf".format(os.environ["HOME"], "LightGCN_filters"),
        # bbox_inches="tight",
    )
    plt.show()
