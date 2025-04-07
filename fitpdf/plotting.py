#
# Plotting related helper functions.
#

import arviz as az
import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from KDEpy import FFTKDE, TreeKDE
from KDEpy.bw_selection import improved_sheather_jones
import xarray as xr

import fitpdf.models as fmodels
from fitpdf.stats import get_adaptive_bandwidth


def plot_chains(idata, params):
    """
    Plot the chains.
    """

    az.plot_trace(idata)

    fig = plt.gcf()
    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "chains.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)


def plot_corner(idata, params):
    """
    Make a corner plot.
    """

    # get maximum likelihood values
    posterior = az.extract(idata.posterior)
    llike = az.extract(idata.log_likelihood)

    max_likelihood_idx = llike.sum("obs_id").argmax()
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

    fig = corner.corner(
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

    # output plot to file
    if params["output"]:
        fig.savefig(
            "corner.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)

    # reset
    matplotlib.rcParams["font.size"] = fontsize_before


def plot_fit(idata, pp, offp, params):
    """
    Plot the distribution fit.
    """

    obs_data = np.sort(idata.observed_data["obs"].values)

    fig = plt.figure()
    ax = fig.add_subplot()

    # plot the observed data
    # kernel density estimate using adaptive bandwidth
    isj_bw = improved_sheather_jones(obs_data.reshape(obs_data.shape[0], -1))
    print(f"ISJ kernel bandwidth: {isj_bw:.5f}")

    bandwidths = get_adaptive_bandwidth(obs_data, min_bw=7.0 * isj_bw)
    print(f"Bandwidths: {bandwidths}")

    kde_x, kde_y = TreeKDE(kernel="gaussian", bw=bandwidths).fit(obs_data).evaluate()

    if params["labels"] is None:
        label = "data"
    else:
        label = params["labels"][0]

    ax.plot(kde_x, kde_y, color="black", lw=2, label=label, zorder=4)

    # rug plot
    # use data coordinates in horizontal and axis coordinates in vertical direction
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.scatter(
        obs_data,
        [0.99 for _ in range(len(obs_data))],
        marker="|",
        color="black",
        lw=0.3,
        transform=trans,
        alpha=0.1,
        rasterized=True,
    )

    # off pulse
    isj_bw = improved_sheather_jones(offp.reshape(offp.shape[0], -1))
    bandwidths = get_adaptive_bandwidth(offp, min_bw=10.0 * isj_bw)
    kde_x, kde_y = TreeKDE(kernel="gaussian", bw=bandwidths).fit(offp).evaluate()

    ax.fill_between(
        x=kde_x,
        y1=kde_y,
        y2=0,
        facecolor="dimgrey",
        edgecolor="none",
        label="off",
        lw=0,
        alpha=0.2,
        zorder=3,
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

        ax.plot(
            kde_x, kde_y, color="C0", lw=0.5, zorder=3.5, alpha=0.1, rasterized=True
        )

    # plot the individual model components
    plot_range = xr.DataArray(
        np.linspace(
            obs_data.min(),
            obs_data.max(),
            num=1000,
        ),
        dims="plot",
    )

    for i in range(2):
        component = i

        ana_full = xr.apply_ufunc(
            fmodels.normal_lognormal_analytic_pdf,
            plot_range,
            idata.posterior["w"],
            idata.posterior["mu"],
            idata.posterior["sigma"],
            component,
        )
        pdf = ana_full.mean(dim=("chain", "draw")).sel(component=i)

        ax.plot(plot_range, pdf, lw=1, label=f"c{i}", zorder=6)

    ax.legend(loc="best", frameon=False)
    if params["title"] is not None:
        ax.set_title(params["title"])
    ax.set_xlabel(r"$F_\mathrm{on} \: / \: \left< F_\mathrm{on} \right>$")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    ax.set_xlim(left=1.25 * obs_data.min(), right=1.05 * obs_data.max())

    # set the limits to a bin count of unity
    _density_data, _ = np.histogram(
        obs_data,
        bins=params["nbin"],
        density=True,
    )

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
