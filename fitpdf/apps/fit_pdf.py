import argparse
import logging
import os
import signal
import sys

# switch between interactive and non-interactive mode
import matplotlib

if "DISPLAY" not in os.environ:
    # set a rendering backend that does not require an X server
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

from spanalysis.general_helpers import (
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

    # check that files exist
    for item in args.files:
        if not os.path.isfile(item):
            log.error(f"File does not exist: {item}")
            sys.exit(1)

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

    dfs = []

    for item in args.files:
        print(f"Processing: {item}")
        df = pd.read_csv(item)
        df["filename"] = item
        dfs.append(df)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
