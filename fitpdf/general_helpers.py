#
#   2025 Fabian Jankowski
#   General helper functions.
#

import logging
import signal
import sys

import matplotlib


def configure_logging(level=logging.INFO):
    """
    Initialize and configure logging.

    Parameters
    ----------
    level: enum
        Requested logging level.
    """

    log = logging.getLogger("fitpdf")
    log.setLevel(logging.DEBUG)
    log.propagate = False

    # ensure a clean root handler
    for h in list(log.handlers):
        log.removeHandler(h)

    console = logging.StreamHandler()
    console.setLevel(level)
    console_formatter = logging.Formatter("%(levelname)s,%(name)s: %(message)s")
    console.setFormatter(console_formatter)
    log.addHandler(console)


def customise_matplotlib_format():
    """
    Customise the matplotlib output formatting.
    """

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 14.0
    matplotlib.rcParams["lines.markersize"] = 8
    matplotlib.rcParams["legend.frameon"] = False
    # make tickmarks more visible
    matplotlib.rcParams["xtick.major.size"] = 8
    matplotlib.rcParams["xtick.major.width"] = 1.5
    matplotlib.rcParams["xtick.minor.size"] = 4
    matplotlib.rcParams["xtick.minor.width"] = 1.5
    matplotlib.rcParams["ytick.major.size"] = 8
    matplotlib.rcParams["ytick.major.width"] = 1.5
    matplotlib.rcParams["ytick.minor.size"] = 4
    matplotlib.rcParams["ytick.minor.width"] = 1.5


def signal_handler(signum, frame):
    """
    Handle unix signals sent to the program.
    """

    log = logging.getLogger("fitpdf.general_helpers")

    # treat SIGINT/INT/CRTL-C
    if signum == signal.SIGINT:
        log.warning("SIGINT received, stopping the program.")
        sys.exit(1)
