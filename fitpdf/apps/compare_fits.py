#
#   2025 Fabian Jankowski
#   Compare fits.
#

import argparse
import logging
import os
import signal
import sys

import arviz as az

if "DISPLAY" not in os.environ:
    # set a rendering backend that does not require an X server
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitpdf.general_helpers import (
    configure_logging,
    signal_handler,
)
from fitpdf.plotting import plot_fit_comparison


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Compare fits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Names of files to process. The input files must be InferenceData files produced by fitpdf-fit.",
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

    log = logging.getLogger("fitpdf.compare_fits")

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
    log = logging.getLogger("fitpdf.compare_fits")

    # handle command line arguments
    args = parse_args()

    # sanity check command line arguments
    check_args(args)

    params = {
        "dpi": 300,
        "output": args.output,
    }

    idatas = {}

    for item in args.files:
        print(f"Processing: {item}")
        _idata = az.from_netcdf(item)
        _label = item.rstrip(".nc")
        _label = _label.lstrip("idata_")
        idatas[_label] = _idata

    df_comp = az.compare(idatas)
    print(df_comp)

    plot_fit_comparison(df_comp, params)

    plt.show()

    log.info("All done.")


if __name__ == "__main__":
    main()
