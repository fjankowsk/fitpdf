#
#
#

import numpy as np
import pymc as pm
from scipy import stats


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

        # priors
        mu = pm.Normal("mu", mu=np.array([0, 1]), sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=np.array([1, 1]))

        # 1) normal distribution for nulling
        # 2) normal distribution for pulses

        components = pm.Normal.dist(
            mu=mu,
            sigma=sigma,
            shape=(2,),
        )

        pm.Mixture("obs", w=w, comp_dists=components, observed=data)

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

    coords = {"component": np.arange(2), "obs_id": np.arange(len(data))}

    with pm.Model(coords=coords) as model:
        x = pm.Data("x", data, dims="obs_id")

        # mixture weights
        w = pm.Dirichlet("w", a=np.array([1, 1]), dims="component")

        # priors
        mu = pm.Normal("mu", mu=np.array([0, 1]), sigma=1, dims="component")
        sigma = pm.HalfNormal("sigma", sigma=np.array([1, 1]), dims="component")

        # 1) normal distribution for nulling
        norm = pm.Normal.dist(mu=mu[0], sigma=sigma[0])

        # 2) lognormal distribution for pulses
        lognorm = pm.Lognormal.dist(mu=mu[1], sigma=sigma[1])

        components = [norm, lognorm]

        pm.Mixture("obs", w=w, comp_dists=components, observed=x, dims="obs_id")

    return model


def normal_lognormal_analytic(x, w, mu, sigma):
    norm = stats.norm.pdf(x, loc=mu[0], scale=sigma[0])
    lognorm = stats.lognorm.pdf(x, s=sigma[1], loc=mu[1])

    pdf = w[0] * norm + w[1] * lognorm

    return pdf


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

        # priors
        mu = pm.Normal("mu", mu=np.array([0, 1]), sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=np.array([1, 1]))

        # 1) lognormal distribution
        lognorm1 = pm.Lognormal.dist(mu=mu[0], sigma=sigma[0])

        # 2) lognormal distribution
        lognorm2 = pm.Lognormal.dist(mu=mu[1], sigma=sigma[1])

        components = [lognorm1, lognorm2]

        pm.Mixture("obs", w=w, comp_dists=components, observed=data)

    return model
