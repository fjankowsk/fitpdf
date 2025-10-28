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
import fitpdf.models as fmodels
from fitpdf.plotting import (
    plot_chains,
    plot_corner,
    plot_fit,
    plot_pedist,
    plot_prior_predictive,
)


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
        "--mean",
        dest="mean",
        type=float,
        metavar=("value"),
        default=None,
        help="The global mean fluence by which to divide the histograms. The default behaviour is to determine it automatically from the on-pulse fluence data.",
    )

    # fit parameter options
    fitp = parser.add_argument_group(title="Fit parameters")

    fitp.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="Enable fast processing. This reduces the number of MCMC samples drastically and is recommended against for publication-quality fits.",
    )

    fitp.add_argument(
        "--model",
        dest="model",
        choices=["NL", "NN", "NNL", "NNP"],
        default="NNL",
        help="Use the specified distribution model, where N denotes a Normal, L a Lognormal, and P a powerlaw (Pareto) component. For instance, the default NNL model consists of two Normal and one Lognormal distributions.",
    )

    fitp.add_argument(
        "--weights",
        dest="weights",
        type=float,
        metavar=("value"),
        nargs="+",
        default=None,
        help="Override the default component distribution weights in the model prior. This is sometimes useful to ensure convergence of the fit. The weights are given as simple floating point numbers (not percentages) and must sum to unity. For instance, [0.2, 0.3, 0.5] assigns an average prior weight of 20, 30, and 50 per cent to each of the component distributions, respectively. The number of weights specified must match the number of model components, e.g. three for the NNL model.",
    )

    # options that affect the output formatting
    output = parser.add_argument_group(title="Output formatting")

    output.add_argument(
        "--label",
        dest="label",
        type=str,
        metavar=("name"),
        default=None,
        help="The label to use for the input file.",
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

    output.add_argument(
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

    # check the mean
    if args.mean is not None:
        if args.mean > 0:
            pass
        else:
            log.error(f"The mean fluence is invalid: {args.mean}")
            sys.exit(1)

    # nbin
    if not args.nbin > 1:
        log.error(f"The number of bins is invalid: {args.nbin}")
        sys.exit(1)

    # weights
    if args.weights is not None:
        if len(args.weights) == len(args.model) and sum(args.weights) == 1.0:
            pass
        else:
            log.error(f"The weights are invalid: {args.weights}")
            sys.exit(1)

    # check that file exist
    if not os.path.isfile(args.filename):
        log.error(f"File does not exist: {args.filename}")
        sys.exit(1)


def get_clean_data(t_df):
    """
    Clean the data.

    Returns
    -------
    fon: ~np.array of float
        The cleaned on-pulse fluence data.
    foff: ~np.array of float
        The cleaned off-pulse fluence data.
    """

    log = logging.getLogger("fitpdf.fit_pdf")

    df = t_df.copy()

    # exclude rfi zapped data
    mask_zapped = df["zapped"].astype(bool)
    mask_good = np.logical_not(mask_zapped)

    # on-pulse
    fon = df["fluence_on"].to_numpy()
    mask = mask_good & np.isfinite(fon)

    log.info(f"Size of Fon before cleaning: {fon.size}")
    fon = fon[mask]
    fon = np.sort(fon)
    log.info(f"Size of Fon after cleaning: {fon.size}")

    # off-pulse
    if df["nbin_off_same"].iat[0] == df["nbin_on"].iat[0]:
        foff = df["fluence_off_same"].to_numpy()
    else:
        foff = (df["fluence_off_same"] * df["nbin_on"] / df["nbin_off_same"]).to_numpy()
        log.warning(
            "Off-pulse phase range is too small: {0}, {1}. Scaling the off-pulse fluences.".format(
                df["nbin_off_same"].iat[0], df["nbin_on"].iat[0]
            )
        )

    log.info(f"Size of Foff before cleaning: {foff.size}")
    foff = foff[mask]
    foff = np.sort(foff)
    log.info(f"Size of Foff after cleaning: {foff.size}")

    assert np.all(np.isfinite(fon))
    assert np.all(np.isfinite(foff))
    assert fon.size == foff.size

    return fon, foff


def fit_pedist(t_data, t_offp, params):
    """
    Fit pulse-energy distribution data.

    Parameters
    ----------
    t_data: ~np.array of float
        The on-pulse data.
    t_offp: ~np.array of float
        The off-pulse data.
    params: dict
        Additional parameters that influence the processing.
    """

    log = logging.getLogger("fitpdf.fit_pdf")

    data = t_data.copy()
    offp = t_offp.copy()

    # model selection
    if params["model"] == "NL":
        mobj = fmodels.NL()
    elif params["model"] == "NN":
        mobj = fmodels.NN()
    elif params["model"] == "NNL":
        mobj = fmodels.NNL()
    elif params["model"] == "NNP":
        log.error("The NNP model is currently disabled.")
        sys.exit(1)
    else:
        raise NotImplementedError("Model not implemented: %s", params["model"])

    model = mobj.get_model(data, offp, params)

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

    plot_fit(mobj, model, idata, offp, params)

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
        _samples = mobj.get_mode(idata.posterior, icomp)
        quantiles = _samples.sel(component=icomp).quantile(
            q=[0.16, 0.5, 0.84], dim=("chain", "draw")
        )
        error = np.maximum(
            np.abs(quantiles[1] - quantiles[0]), np.abs(quantiles[2] - quantiles[1])
        )
        mode = {"value": quantiles[1], "error": error}

        print(
            "Component {0}: {1:.3f} +- {2:.3f} %, {3:.3f} +- {4:.3f}".format(
                icomp,
                100.0 * weight["value"],
                100.0 * weight["error"],
                mode["value"],
                mode["error"],
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
        "dpi": 300,
        "fast": args.fast,
        "label": args.label,
        "mean": args.mean,
        "model": args.model,
        "nbin": args.nbin,
        "output": args.output,
        "publish": False,
        "title": args.title,
        "weights": args.weights,
    }

    print(f"Processing: {args.filename}")
    df = pd.read_csv(args.filename)
    df["filename"] = args.filename

    # sanity check
    _fields = [
        "rotation",
        "zapped",
        "fluence_on",
        "nbin_on",
        "fluence_off",
        "nbin_off",
        "fluence_off_same",
        "nbin_off_same",
    ]
    assert all(item in df.columns for item in _fields)

    _fon, _foff = get_clean_data(df)

    for item, _label in zip([_fon, _foff], ["On", "Off"]):
        print(
            "{0}-pulse mean, median, std, number of samples: {1:.5f}, {2:.5f}, {3:.5f}, {4}".format(
                _label, np.mean(item), np.median(item), np.std(item), len(item)
            )
        )

    # standardise the data by dividing by the mean on-pulse fluence
    # we should also divide by the on-pulse standard deviation for a
    # proper data standardisation (unity mean and unity standard deviation)
    # however, then we must keep track of the parameters and transform the inference data back later
    # we express the priors in data units (on and off-pulse mean fluence) instead here
    # this avoids the back transformation later
    if args.mean is None:
        _global_mean = np.mean(_fon)
    else:
        _global_mean = params["mean"]

    print(f"Global mean on-pulse fluence: {_global_mean}")

    params["outfile"] = "pedist_pdf.pdf"
    plot_pedist(_fon / _global_mean, _foff / _global_mean, params)

    fit_pedist(_fon / _global_mean, _foff / _global_mean, params)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
