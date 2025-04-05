#
# Plotting related helper functions.
#

import arviz as az
import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from KDEpy import FFTKDE

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

    obs_data = idata.observed_data["obs"].values

    fig = plt.figure()
    ax = fig.add_subplot()

    # plot the observed data
    _density_data, _, _ = ax.hist(
        obs_data,
        bins=params["nbin"],
        color="black",
        density=True,
        histtype="step",
        linewidth=2,
        label="data",
        zorder=4,
    )

    # rug plot
    # use data coordinates in horizontal and axis coordinates in vertical direction
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.scatter(
        obs_data,
        [0.99 for _ in range(len(obs_data))],
        marker="|",
        color="black",
        lw=0.8,
        transform=trans,
        alpha=0.3,
    )

    # plot the mean model
    samples = pp.posterior_predictive["obs"].values.reshape(-1)
    kde_x, kde_y = FFTKDE(kernel="gaussian", bw="ISJ").fit(samples).evaluate()

    ax.plot(kde_x, kde_y, color="firebrick", lw=1.5, label="model", zorder=5)

    # plot the individual pp draws
    _ndraw = 50
    rng = np.random.default_rng()
    idxs_chain = rng.integers(
        low=0, high=len(pp.posterior_predictive["chain"]), size=_ndraw
    )
    idxs_draw = rng.integers(
        low=0, high=len(pp.posterior_predictive["draw"]), size=_ndraw
    )

    for ichain, idraw in zip(idxs_chain, idxs_draw):
        samples = pp.posterior_predictive["obs"].isel(chain=ichain, draw=idraw).values
        kde_x, kde_y = FFTKDE(kernel="gaussian", bw="ISJ").fit(samples).evaluate()

        ax.plot(kde_x, kde_y, color="C0", lw=0.5, zorder=3, alpha=0.1)

    # plot the individual model components
    mean_params = idata.posterior.mean(dim=("chain", "draw"))
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

        ax.plot(plot_range, analytic, label=f"c{i}", zorder=6)

    ax.legend(loc="best", frameon=False)
    if params["title"] is not None:
        ax.set_title(params["title"])
    ax.set_xlabel(r"$F_\mathrm{on} \: / \: \left< F_\mathrm{on} \right>$")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    ax.set_xlim(left=1.25 * obs_data.min(), right=1.05 * obs_data.max())

    # set the limits to a bin count of unity
    _mask = np.isfinite(_density_data) & (_density_data > 0)
    min_density = np.min(_density_data[_mask])
    max_density = np.max(_density_data[_mask])

    ax.set_ylim(bottom=0.7 * min_density, top=2.0 * max_density)

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "pedist_fit.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)
