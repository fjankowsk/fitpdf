#
#   2025 Fabian Jankowski
#   Fit measured distribution data.
#

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
import numpy as np
import pandas as pd
import pymc as pm

from fitpdf.general_helpers import (
    configure_logging,
    customise_matplotlib_format,
    signal_handler,
)
from spanalysis.apps.plot_dist import plot_pe_dist
import fitpdf.models as fmodels
from fitpdf.plotting import plot_chains, plot_corner, plot_fit, plot_prior_predictive


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
        "filename",
        type=str,
        help="Name of file to process. The input file must be produced by the fluence time series option of plot-profilestack.",
    )

    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="Enable fast processing. This reduces the number of MCMC steps drastically.",
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
        default=-3.0,
        help="Ignore fluence data below this mean fluence threshold, i.e. select only data where fluence / mean > meanthresh.",
    )

    parser.add_argument(
        "--model",
        dest="model",
        choices=["NL", "NN", "NNL"],
        default="NNL",
        help="Use the specified distribution model, where N denotes a Normal and L a Lognormal component. For instance, the default NNL model consists of two Normal and one Lognormal distributions.",
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

    # check that file exist
    if not os.path.isfile(args.filename):
        log.error(f"File does not exist: {args.filename}")
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

    # model selection
    if params["model"] == "NL":
        mobj = fmodels.NL()
    elif params["model"] == "NN":
        mobj = fmodels.NN()
    elif params["model"] == "NNL":
        mobj = fmodels.NNL()
    else:
        raise NotImplementedError("Model not implemented: %s", params["model"])

    model = mobj.get_model(data, offp)

    print(f"All RVs: {model.basic_RVs}")
    print(f"Free RVs: {model.free_RVs}")
    print(f"Observed RVs: {model.observed_RVs}")
    print(f"Initial point: {model.initial_point()}")

    config = {}

    if params["fast"]:
        config["draws"] = 700
        config["tune"] = 700
    else:
        config["draws"] = 10000
        config["tune"] = 2000

    with model:
        idata = pm.sample(
            draws=config["draws"],
            tune=config["tune"],
            chains=4,
            init="advi+adapt_diag",
            nuts={"target_accept": 0.9},
        )
        pm.compute_log_likelihood(idata)

    _df_result = az.summary(idata)
    print(_df_result)
    _df_result.to_csv("fit_result.csv")

    plot_chains(idata, params)
    plot_corner(idata, params)

    # compute prior predictive samples
    with model:
        # sample all the parameters
        pp = pm.sample_prior_predictive()

    assert hasattr(pp, "prior_predictive")

    plot_prior_predictive(pp, data, offp, params)

    # compute posterior predictive samples
    thinned_idata = idata.sel(draw=slice(None, None, 20))

    with model:
        pp = pm.sample_posterior_predictive(thinned_idata, var_names=["obs"])
        idata.extend(pp)

    assert hasattr(idata, "posterior_predictive")

    # save idata to file
    _filename = "idata_{0}.nc".format(params["model"])
    az.to_netcdf(idata, _filename)

    plot_fit(mobj, idata, offp, params)

    # output the fit parameters
    print("\nFit parameters")
    for icomp in range(mobj.ncomp):
        # weight
        _samples = idata.posterior["w"]
        quantiles = _samples.sel(component=icomp).quantile(
            q=[0.16, 0.5, 0.84], dim=("chain", "draw")
        )
        error = np.maximum(
            np.abs(quantiles[1] - quantiles[0]), np.abs(quantiles[2] - quantiles[1])
        )
        weight = {"value": quantiles[1], "error": error}

        # mode
        _samples = mobj.get_mode(idata.posterior["mu"], idata.posterior["sigma"], icomp)
        quantiles = _samples.sel(component=icomp).quantile(
            q=[0.16, 0.5, 0.84], dim=("chain", "draw")
        )
        error = np.maximum(
            np.abs(quantiles[1] - quantiles[0]), np.abs(quantiles[2] - quantiles[1])
        )
        mode = {"value": quantiles[1], "error": error}

        print(
            "Component {0}: {1:.3f} +- {2:.3f}, {3:.3f} +- {4:.3f}".format(
                icomp, weight["value"], weight["error"], mode["value"], mode["error"]
            )
        )


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
        "fast": args.fast,
        "labels": args.labels,
        "log": args.log,
        "mean": args.mean,
        "mean_thresh": args.mean_thresh,
        "model": args.model,
        "nbin": args.nbin,
        "output": args.output,
        "publish": False,
        "title": args.title,
    }

    print(f"Processing: {args.filename}")
    df = pd.read_csv(args.filename)
    df["filename"] = args.filename

    _data, _offp = plot_pe_dist([df], params)

    fit_pe_dist(_data / params["mean"], _offp / params["mean"], params)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
