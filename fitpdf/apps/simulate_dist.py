#
#   2025 Fabian Jankowski
#   Simulate distributions.
#

import argparse
import logging
import os
import signal
import sys

import numpy as np
import pandas as pd
import pymc as pm

import matplotlib

if "DISPLAY" not in os.environ:
    # set a rendering backend that does not require an X server
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitpdf.general_helpers import (
    configure_logging,
    customise_matplotlib_format,
    signal_handler,
)
from fitpdf.plotting import plot_pedist


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Simulate distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--nsamp",
        dest="nsamp",
        type=int,
        metavar=("value"),
        default=10000,
        help="Number of random samples to draw from the simulated distribution.",
    )

    parser.add_argument(
        "--randomseed",
        dest="randomseed",
        type=int,
        metavar=("value"),
        default=None,
        help="Enable deterministic mode by providing a seed value for the random number generator. This is useful if you want to fix the underlying distribution when changing the number of samples. The default behaviour is non-deterministic, i.e. the simulation uses different distribution parameters in each run.",
    )

    # options that affect the output formatting
    output = parser.add_argument_group(title="Output formatting")

    output.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store_true",
        default=False,
        help="Output plots to file rather than to screen.",
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

    log = logging.getLogger("fitpdf.simulate_dist")

    # nsamp
    if not args.nsamp > 100:
        log.error(f"Number of samples is invalid: {args.nsamp}")
        sys.exit(1)

    # randomseed
    if args.randomseed is not None:
        if not args.randomseed > 0:
            log.error(f"Random seed value is invalid: {args.randomseed}")
            sys.exit(1)


#
# MAIN
#


def main():
    # start signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # set up logging
    configure_logging()
    log = logging.getLogger("fitpdf.simulate_dist")

    # handle command line arguments
    args = parse_args()

    # sanity check command line arguments
    check_args(args)

    # tweak the matplotlib output formatting
    customise_matplotlib_format()

    params = {
        "dpi": 300,
        "nbin": "auto",
        "nsamp": args.nsamp,
        "output": args.output,
        "title": None,
    }

    # switch between non-deterministic (default) and deterministic mode
    if args.randomseed is None:
        rng = None
    else:
        rng = np.random.default_rng(args.randomseed)

    weights = [0.2, 0.3, 0.5]
    weights /= np.sum(weights)
    mu = [0.0, 0.5, 0.0]
    sigma = [2.0, 1.0, 0.5]

    print(f"Mixture weights: {weights}")

    # off-pulse
    foff = pm.Normal.dist(mu=mu[0], sigma=sigma[0])

    foff_samples = pm.draw(foff, draws=params["nsamp"], random_seed=rng)

    # on-pulse
    fon = pm.Mixture.dist(
        w=weights,
        comp_dists=[
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1]),
            pm.LogNormal.dist(mu=mu[2], sigma=sigma[2]),
        ],
    )

    fon_samples = pm.draw(fon, draws=params["nsamp"], random_seed=rng)

    # write to disk
    _temp = {
        "rotation": np.arange(params["nsamp"]),
        "zapped": np.zeros(params["nsamp"]).astype(int),
        "fluence_on": fon_samples,
        "nbin_on": 64,
        "fluence_off": foff_samples,
        "nbin_off": 64,
        "fluence_off_same": foff_samples,
        "nbin_off_same": 64,
    }
    _df = pd.DataFrame(_temp)
    _df.to_csv("simulated_fluences.csv", index=False)

    params["outfile"] = "simulated_pdf.pdf"
    plot_pedist(fon_samples, foff_samples, params)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
