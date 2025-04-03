#
#
#

import numpy as np
import pymc as pm


def normal_normal(t_data):
    """
    Construct a normal - normal mixture model.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()

    with pm.Model() as model:
        # mixture weights
        w = pm.Dirichlet("w", a=np.array([1, 1]))

        # 1) normal distribution for nulling
        mu1 = pm.Normal("mu1", mu=0, sigma=1)
        sigma1 = pm.HalfNormal("sigma1", sigma=1)

        # 2) normal distribution for pulses
        mu2 = pm.Normal("mu2", mu=1, sigma=1)
        sigma2 = pm.HalfNormal("sigma2", sigma=1)

        components = pm.Normal.dist(
            mu=pm.math.stack([mu1, mu2]),
            sigma=pm.math.stack([sigma1, sigma2]),
            shape=(2,),
        )

        pm.Mixture("likelihood", w=w, comp_dists=components, observed=data)

    return model


def normal_lognormal(t_data):
    """
    Construct a normal - lognormal mixture model.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()

    with pm.Model() as model:
        # mixture weights
        w = pm.Dirichlet("w", a=np.array([1, 1]))

        # 1) normal distribution for nulling
        mu1 = pm.Normal("mu1", mu=0, sigma=1)
        sigma1 = pm.HalfNormal("sigma1", sigma=1)

        norm = pm.Normal.dist(mu=mu1, sigma=sigma1)

        # 2) lognormal distribution for pulses
        mu2 = pm.Normal("mu2", mu=1, sigma=1)
        sigma2 = pm.HalfNormal("sigma2", sigma=1)

        lognorm = pm.Lognormal.dist(mu=mu2, sigma=sigma2)

        components = [norm, lognorm]

        pm.Mixture("likelihood", w=w, comp_dists=components, observed=data)

    return model


def lognormal_lognormal(t_data):
    """
    Construct a lognormal - lognormal mixture model.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()

    with pm.Model() as model:
        # mixture weights
        w = pm.Dirichlet("w", a=np.array([1, 1]))

        # 1) lognormal distribution
        mu1 = pm.Normal("mu1", mu=0, sigma=1)
        sigma1 = pm.HalfNormal("sigma1", sigma=1)

        lognorm1 = pm.Lognormal.dist(mu=mu1, sigma=sigma1)

        # 2) lognormal distribution
        mu2 = pm.Normal("mu2", mu=1, sigma=1)
        sigma2 = pm.HalfNormal("sigma2", sigma=1)
        lognorm2 = pm.Lognormal.dist(mu=mu2, sigma=sigma2)

        components = [lognorm1, lognorm2]

        pm.Mixture("likelihood", w=w, comp_dists=components, observed=data)

    return model
