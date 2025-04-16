#
#   2025 Fabian Jankowski
#   Statistics related helper functions.
#

import numpy as np


def get_adaptive_bandwidth(t_data, min_bw):
    """
    Compute an adaptive bandwidth for kernel density estimation (KDE).

    The bandwidth changes based on the density of data points.

    Parameters
    ----------
    t_data: ~np.array
        The input data.
    min_bw: float
        The minimum bandwidth to enforce.

    Returns
    -------
    bandwidths: ~np.array
        The kernel bandwidth for each data point.
    """

    data = t_data.copy()
    data = np.sort(data)

    assert min_bw > 0

    bandwidths = np.zeros(len(data))
    bandwidths[:-1] = 0.5 * np.diff(data)
    bandwidths[-1] = bandwidths[-2]
    bandwidths = np.clip(bandwidths, a_min=min_bw, a_max=None)

    return bandwidths
