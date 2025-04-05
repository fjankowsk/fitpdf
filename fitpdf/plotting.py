#
# Plotting related helper functions.
#

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

import fitpdf.models as fmodels


def plot_corner(idata):
    """
    Make a corner plot.
    """

    corner.corner(idata, plot_datapoints=False)


def plot_fit(data, idata, pp, params):
    """
    Plot the distribution fit.
    """

    figsize = (6.4, 4.8)
    ax = az.plot_ppc(pp, figsize=figsize, num_pp_samples=50)

    mean_params = idata.posterior.mean(("chain", "draw"))

    # plot the individual components
    plot_range = np.linspace(data.min(), data.max(), num=500)

    for i in range(2):
        new_w = np.zeros(2)
        new_w[i] = mean_params["w"][i].values

        analytic = fmodels.normal_lognormal_analytic(
            plot_range, new_w, mean_params["mu"].values, mean_params["sigma"].values
        )

        ax.plot(plot_range, analytic, label=f"c{i}")

    ax.legend([])
    ax.get_legend().remove()
    ax.set_xlabel(r"$F_\mathrm{on} \: / \: \left< F_\mathrm{on} \right>$")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    # set ylim to the density corresponding to one bin count
    # ax.set_ylim(bottom=0.7 * min_density)
    ax.set_ylim(bottom=1.0e-6, top=10.0)

    fig = plt.gcf()
    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "pedist_fit.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)
