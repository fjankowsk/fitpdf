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

if "DISPLAY" not in os.environ:
    # set a rendering backend that does not require an X server
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitpdf.general_helpers import (
    configure_logging,
    customise_matplotlib_format,
    signal_handler,
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
        "nsamp": args.nsamp,
        "output": args.output,
    }

    weights = [0.3, 0.3, 0.7]
    mu = [0.0, 0.3, np.log(1.75)]
    sigma = [2.0, 1.0, 1.0]

    # off-pulse
    foff = pm.Normal.dist(mu=mu[0], sigma=sigma[0])

    foff_samples = pm.draw(foff, draws=params["nsamp"])

    # on-pulse
    fon = pm.Mixture.dist(
        w=weights,
        comp_dists=[
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1]),
            pm.LogNormal.dist(mu=mu[2], sigma=sigma[2]),
        ],
    )

    fon_samples = pm.draw(fon, draws=params["nsamp"])

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

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        foff_samples,
        bins="auto",
        color="dimgrey",
        density=True,
        histtype="stepfilled",
        zorder=3,
        alpha=0.3,
    )

    ax.hist(
        fon_samples, bins="auto", color="black", density=True, histtype="step", zorder=5
    )

    ax.set_xlabel(r"$F \: / \: \left< F_\mathrm{on} \right>$")
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    fig.tight_layout()

    # output plot to file
    if params["output"]:
        fig.savefig(
            "simulated_pdf.pdf",
            bbox_inches="tight",
            dpi=params["dpi"],
        )

        plt.close(fig)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
