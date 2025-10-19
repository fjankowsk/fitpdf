#
#   2025 Fabian Jankowski
#   Plotting related helper functions.
#

import arviz as az
import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from KDEpy import FFTKDE, TreeKDE
from KDEpy.bw_selection import improved_sheather_jones
from scipy import stats
import xarray as xr

from fitpdf.stats import get_adaptive_bandwidth


def plot_chains(idata, params):
    """
    Plot the chains.

    Parameters
    ----------
    idata: ~az.InterferenceData
        The input data.
    params: dict
        Additional parameters that influence the processing.
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

    Parameters
    ----------
    idata: ~az.InterferenceData
        The input data.
    params: dict
        Additional parameters that influence the processing.
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


def plot_adaptive_bandwidths(t_data, t_bandwidths, params):
    """
    Visualise the KDE adaptive bandwidths.

    Parameters
    ----------
    t_data: ~np.array of float
        The input data of the KDE.
    t_bandwidths: ~np.array of float
        The adaptive KDE bandwidths for each data point.
    params: dict
        Other parameters that influence the processing.
    """

    data = t_data.copy()
    bandwidths = t_bandwidths.copy()

    ys = np.arange(len(data)) / len(data)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(data, ys, marker="+", color="black", label="data", zorder=5)

    ax.errorbar(
        x=data,
        y=ys,
        xerr=bandwidths,
        color="C1",
        elinewidth=0.5,
        label="bandwidth",
        linestyle="none",
        marker=".",
        zorder=4,
    )

    ax.legend(loc="best", frameon=False)
    ax.set_xlabel("Value")

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "kde_adaptive_bandwidths.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)


def plot_fit(mobj, idata, offp, params):
    """
    Plot the distribution fit.

    Parameters
    ----------
    mobj: ~fitpdf.models.Model
        Model object.
    idata: ~az.InterferenceData
        The input data.
    offp: ~pd.DataFrame
        The off-pulse data.
    params: dict
        Additional parameters that influence the processing.
    """

    obs_data = np.sort(idata.observed_data["obs"].values)

    fig = plt.figure()
    ax = fig.add_subplot()

    # plot the observed data
    # kernel density estimate using adaptive bandwidth
    isj_bw_data = improved_sheather_jones(obs_data.reshape(obs_data.shape[0], -1))
    print(f"ISJ kernel bandwidth data: {isj_bw_data:.5f}")

    min_bw_data = 5.0 * isj_bw_data
    print(f"Minimum bandwidth data: {min_bw_data:.5f}")

    bandwidths = get_adaptive_bandwidth(obs_data, min_bw=min_bw_data)
    print(f"Bandwidths: {bandwidths}")
    plot_adaptive_bandwidths(obs_data, bandwidths, params)

    kde_x_data, kde_y_data = (
        TreeKDE(kernel="gaussian", bw=bandwidths).fit(obs_data).evaluate()
    )

    if params["labels"] is None:
        label = "data"
    else:
        label = params["labels"][0]

    ax.plot(kde_x_data, kde_y_data, color="black", lw=2, label=label, zorder=4)

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
    _isj_bw = improved_sheather_jones(offp.reshape(offp.shape[0], -1))
    _bandwidths = get_adaptive_bandwidth(offp, min_bw=5.0 * _isj_bw)
    kde_x, kde_y = TreeKDE(kernel="gaussian", bw=_bandwidths).fit(offp).evaluate()

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

    # plot the best-fitting model
    # use fftkde here with non-adaptive bandwidth for speed
    samples = idata.posterior_predictive["obs"].values.reshape(-1)
    _mask = (samples >= kde_x_data.min()) & (samples <= kde_x_data.max())
    kde_y = (
        FFTKDE(kernel="gaussian", bw=min_bw_data)
        .fit(samples[_mask])
        .evaluate(kde_x_data)
    )

    ax.plot(kde_x_data, kde_y, color="firebrick", lw=1.5, label="model", zorder=5)

    # perform a two-sample kolmogorov-smirnov test
    ks_test = stats.ks_2samp(obs_data, samples, axis=None)
    print("Two-sample KS test data vs model:")
    print(f"Statistic: {ks_test.statistic:.3f}")
    print(f"p-value: {ks_test.pvalue:.3f}")
    print(f"Location: {ks_test.statistic_location:.3f}")
    print(f"Sign: {ks_test.statistic_sign}")

    # plot the individual pp draws
    _ndraw = 50
    rng = np.random.default_rng()
    idxs_chain = rng.integers(
        low=0, high=len(idata.posterior_predictive["chain"]), size=_ndraw
    )
    idxs_draw = rng.integers(
        low=0, high=len(idata.posterior_predictive["draw"]), size=_ndraw
    )

    for ichain, idraw in zip(idxs_chain, idxs_draw):
        samples = (
            idata.posterior_predictive["obs"].isel(chain=ichain, draw=idraw).values
        )
        _bandwidths = get_adaptive_bandwidth(samples, min_bw=min_bw_data)
        kde_y = (
            TreeKDE(kernel="gaussian", bw=_bandwidths).fit(samples).evaluate(kde_x_data)
        )

        ax.plot(
            kde_x_data,
            kde_y,
            color="firebrick",
            lw=0.5,
            zorder=3.5,
            alpha=0.1,
            rasterized=True,
        )

    # plot the individual model components
    plot_range = xr.DataArray(
        np.linspace(
            obs_data.min(),
            obs_data.max(),
            num=500,
        ),
        dims="plot",
    )

    assert hasattr(mobj, "ncomp")

    # plot component pdfs
    for icomp in range(mobj.ncomp):
        _ana_full = xr.apply_ufunc(
            mobj.get_analytic_pdf,
            plot_range,
            idata.posterior["w"],
            idata.posterior["mu"],
            idata.posterior["sigma"],
            icomp,
        )
        _pdf = _ana_full.sel(component=icomp).mean(dim=("chain", "draw"))

        ax.plot(plot_range, _pdf, label=f"c{icomp}", lw=1, zorder=6)

    ax.legend(loc="best", frameon=False)
    if params["title"] is not None:
        ax.set_title(params["title"])
    ax.set_xlabel(r"$F \: / \: \left< F_\mathrm{on} \right>$")
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


def plot_prior_predictive(idata, t_data, t_offp, params):
    """
    Plot the prior predictive samples.

    Parameters
    ----------
    idata: ~az.InterferenceData
        The input data.
    offp: ~pd.DataFrame
        The off-pulse data.
    params: dict
        Additional parameters that influence the processing.
    """

    data = t_data.copy()
    offp = t_offp.copy()

    # 1) density plot for all parameters
    az.plot_density(
        idata,
        group="prior",
        shade=0.1,
    )

    fig = plt.gcf()
    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "prior_predictive_densities.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)

    # 2) comparison with the data
    fig = plt.figure()
    ax = fig.add_subplot()

    bins = np.linspace(data.min(), data.max(), num=params["nbin"])

    # data
    if params["labels"] is None:
        label = "on"
    else:
        label = params["labels"][0]

    ax.hist(
        data,
        bins=bins,
        color="black",
        density=True,
        histtype="step",
        label=label,
        lw=2,
        zorder=4,
    )

    # off-pulse
    ax.hist(
        offp,
        bins=params["nbin"],
        color="dimgrey",
        density=True,
        histtype="stepfilled",
        linewidth=2,
        label="off",
        zorder=3,
        alpha=0.4,
    )

    # prior predictive samples
    ax.hist(
        idata.prior_predictive["obs"].values.reshape(-1),
        bins=bins,
        color="C1",
        density=True,
        histtype="step",
        label="prior predictive",
        lw=2,
        zorder=5,
    )

    ax.legend(loc="best", frameon=False)
    ax.set_xlabel(r"$F \: / \: \left< F_\mathrm{on} \right>$")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "prior_predictive_data.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)


def plot_fit_comparison(df_comp, params):
    """
    Plot a fit comparison.

    Parameters
    ----------
    df_comp: ~pd.DataFrame
        The fit comparison data.
    params: dict
        Additional parameters that influence the processing.
    """

    figsize = (7.4, len(df_comp.index))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    az.plot_compare(df_comp, ax=ax, insample_dev=True, legend=True, plot_ic_diff=True)

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "fit_comparison.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)
