#
# Plotting related helper functions.
#

import arviz as az
import corner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import fitpdf.models as fmodels


def plot_corner(idata, params):
    """
    Make a corner plot.
    """

    # get maximum likelihood values
    posterior = az.extract(idata.posterior)
    llike = az.extract(idata.log_likelihood)

    max_likelihood_idx = llike.sum("obs_dim_0").argmax()
    max_likelihood_idx = max_likelihood_idx["obs"].values
    max_likelihood_values = posterior.isel(sample=max_likelihood_idx)

    # defaults
    bins = 40
    fontsize_before = matplotlib.rcParams["font.size"]
    hist_kwargs = None
    labelpad = 0.125
    max_n_ticks = 5
    plot_datapoints = False
    show_titles = True
    smooth = False

    if params["publish"]:
        hist_kwargs = {"lw": 2.0}
        labelpad = 0.475
        max_n_ticks = 2
        matplotlib.rcParams["font.size"] = 34.0
        show_titles = False
        smooth = True

    corner.corner(
        idata,
        bins=bins,
        hist_kwargs=hist_kwargs,
        labelpad=labelpad,
        max_n_ticks=max_n_ticks,
        truths=max_likelihood_values,
        plot_datapoints=plot_datapoints,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=show_titles,
        smooth=smooth,
        title_kwargs={"fontsize": 10},
    )

    # reset
    matplotlib.rcParams["font.size"] = fontsize_before


def plot_fit(idata, pp, params):
    """
    Plot the distribution fit.
    """

    figsize = (6.4, 4.8)
    ax = az.plot_ppc(pp, figsize=figsize, num_pp_samples=50)

    # plot the individual components
    mean_params = idata.posterior.mean(("chain", "draw"))
    obs_data = idata.observed_data["obs"].values
    plot_range = np.linspace(
        obs_data.min(),
        obs_data.max(),
        num=500,
    )

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
