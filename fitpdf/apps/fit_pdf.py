import argparse
import logging
import os
import signal
import sys

import arviz as az

# switch between interactive and non-interactive mode
import matplotlib

if "DISPLAY" not in os.environ:
    # set a rendering backend that does not require an X server
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pymc as pm

from spanalysis.general_helpers import (
    configure_logging,
    customise_matplotlib_format,
    signal_handler,
)
import fitpdf.models as fmodels
from fitpdf.plotting import plot_chains, plot_corner, plot_fit


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Fit distribution data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Names of files to process. The input files must be produced by the fluence time series option of plot-profilestack.",
    )

    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        nargs="+",
        metavar=("name"),
        default=None,
        help="The labels to use for each input file.",
    )

    parser.add_argument(
        "--mean",
        dest="mean",
        type=float,
        metavar=("value"),
        default=1.0,
        help="The global mean fluence to divide the histograms by.",
    )

    parser.add_argument(
        "--meanthresh",
        dest="mean_thresh",
        type=float,
        metavar=("value"),
        default=-1.0,
        help="Ignore fluence data below this mean fluence threshold, i.e. select only data where fluence / mean > meanthresh.",
    )

    # options that affect the output formatting
    output = parser.add_argument_group(title="Output formatting")

    output.add_argument(
        "--ccdf",
        dest="ccdf",
        action="store_true",
        default=False,
        help="Show the CCDF (cumulative counts) instead of the PDF (differential counts).",
    )

    output.add_argument(
        "--log",
        dest="log",
        action="store_true",
        default=False,
        help="Show histograms in double logarithmic scale.",
    )

    output.add_argument(
        "--nbin",
        dest="nbin",
        type=int,
        metavar=("value"),
        default=50,
        help="The number of histogram bins to use.",
    )

    output.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store_true",
        default=False,
        help="Output plots to file rather than to screen.",
    )

    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        metavar=("text"),
        default=None,
        help="Set a custom figure title.",
    )

    args = parser.parse_args()

    return args


def check_args(args):
    """
    Sanity check the commandline arguments.

    Parameters
    ----------
    args: populated namespace
        The commandline arguments.
    """

    log = logging.getLogger("fitpdf.fit_pdf")

    # check the labels
    if args.labels is not None:
        if len(args.labels) == len(args.files):
            pass
        else:
            log.error(
                "The number of labels is invalid: {0}, {1}".format(
                    len(args.files), len(args.labels)
                )
            )
            sys.exit(1)

    # check the mean
    if args.mean > 0:
        pass
    else:
        log.error(f"The mean fluence is invalid: {args.mean}")
        sys.exit(1)

    # check that files exist
    for item in args.files:
        if not os.path.isfile(item):
            log.error(f"File does not exist: {item}")
            sys.exit(1)


def fit_pe_dist(t_data, t_offp, params):
    """
    Fit pulse-energy distribution.

    Parameters
    ----------
    t_data: ~np.array of float
        The input data.
    t_offp: ~np.array of float
        The off-pulse data.
    params: dict
        Additional parameters that influence the processing.
    """

    data = t_data.copy()
    offp = t_offp.copy()

    model = fmodels.normal_lognormal(data)

    with model:
        idata = pm.sample(draws=10000, chains=4)
        pm.compute_log_likelihood(idata)

    print(az.summary(idata))
    plot_chains(idata, params)
    plot_corner(idata, params)

    # posterior predictive
    thinned_idata = idata.sel(draw=slice(None, None, 20))

    with model:
        pp = pm.sample_posterior_predictive(thinned_idata, var_names=["obs"])

    plot_fit(idata, pp, offp, params)


def plot_pe_dist(dfs, params):
    """
    Plot pulse-energy distributions.

    Parameters
    ----------
    dfs: list of ~pd.DataFrame
        The input data.
    params: dict
        Additional parameters that influence the plotting.
    """

    log = logging.getLogger("fitpdf.fit_pdf")

    fig = plt.figure()
    ax = fig.add_subplot()

    min_density = 1.0e9

    for i, df in enumerate(dfs):
        # use only the good data
        mask_zapped = df["zapped"].astype(bool)
        mask_good = np.logical_not(mask_zapped)

        data = (df["fluence_on"] / df["nbin_on"]).to_numpy()
        log.info(f"File, size data: {i}, {data.size}")

        mask = (
            mask_good
            & np.isfinite(data)
            & (data / params["mean"] > params["mean_thresh"])
        )
        data = data[mask]
        data = np.sort(data)
        log.info(f"File, size good-only: {i}, {data.size}")

        print(
            "File, data mean, median, number of samples: {0}, {1:.5f}, {2:.5f}, {3}".format(
                i, np.mean(data), np.median(data), len(data)
            )
        )

        if params["labels"] is None:
            label = f"{i}"
        else:
            label = params["labels"][i]

        if params["log"]:
            bins = np.geomspace(
                0.1, np.max(data / params["mean"]) + 0.1, num=params["nbin"]
            )
        else:
            bins = params["nbin"]

        if params["ccdf"]:
            cumulative = -1
        else:
            cumulative = False

        # make the first dist always black
        if i == 0:
            color = "black"
        else:
            color = f"C{i - 1}"

        # on-pulse
        _density, _, _ = ax.hist(
            data / params["mean"],
            bins=bins,
            color=color,
            density=True,
            cumulative=cumulative,
            histtype="step",
            linewidth=2,
            label=label,
        )
        _mask = np.isfinite(_density) & (_density > 0)
        if np.min(_density[_mask]) < min_density:
            min_density = np.min(_density[_mask])

        # rug plot
        # use data coordinates in horizontal and axis coordinates in vertical direction
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.scatter(
            data / params["mean"],
            [0.99 for _ in range(len(data))],
            marker="|",
            color=color,
            lw=0.8,
            transform=trans,
            alpha=0.3,
        )

        # show the off-pulse fluence only for the first entry
        if i == 0:
            # off-pulse
            offp = (df["fluence_off"] / df["nbin_off"]).to_numpy()
            mask = (
                mask_good
                & np.isfinite(offp)
                & (offp / params["mean"] > params["mean_thresh"])
            )
            offp = offp[mask]
            offp = np.sort(offp)

            if params["log"]:
                bins = np.geomspace(
                    0.1, np.max(offp / params["mean"]) + 0.1, num=params["nbin"]
                )
            else:
                bins = params["nbin"]

            ax.hist(
                offp / params["mean"],
                bins=bins,
                color="dimgrey",
                density=True,
                cumulative=cumulative,
                histtype="stepfilled",
                linewidth=2,
                label="off",
                zorder=3,
                alpha=0.4,
            )

    # fit data
    fit_pe_dist(data / params["mean"], offp / params["mean"], params)

    ax.legend(loc="best", frameon=False)
    if params["title"] is not None:
        ax.set_title(params["title"])
    ax.set_xlabel(r"$F_\mathrm{on} \: / \: \left< F_\mathrm{on} \right>$")
    if params["ccdf"]:
        ylabel = "CCDF"
    else:
        ylabel = "PDF"
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    if params["log"]:
        ax.set_xscale("log")

    # set ylim to the density corresponding to one bin count
    ax.set_ylim(bottom=0.7 * min_density)

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "pedist_pdf.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)


#
# MAIN
#


def main():
    # start signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # set up logging
    configure_logging()
    log = logging.getLogger("fitpdf.fit_pdf")

    # handle command line arguments
    args = parse_args()

    # sanity check command line arguments
    check_args(args)

    # tweak the matplotlib output formatting
    customise_matplotlib_format()

    params = {
        "ccdf": args.ccdf,
        "dpi": 300,
        "labels": args.labels,
        "log": args.log,
        "mean": args.mean,
        "mean_thresh": args.mean_thresh,
        "nbin": args.nbin,
        "output": args.output,
        "publish": False,
        "title": args.title,
    }

    dfs = []

    for item in args.files:
        print(f"Processing: {item}")
        df = pd.read_csv(item)
        df["filename"] = item
        dfs.append(df)

    plot_pe_dist(dfs, params)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
