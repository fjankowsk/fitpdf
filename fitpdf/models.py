#
#   2025 Fabian Jankowski
#   Distribution models.
#

import numpy as np
import pymc as pm


def normal_normal(t_data, t_offp):
    """
    Construct a normal - normal mixture model.

    Parameters
    ----------
    t_data: ~np.array of float
        The input data to be fit.
    t_offp: ~np.array of float
        The off-pulse data.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()
    offp = t_offp.copy()

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


def normal_lognormal(t_data, t_offp):
    """
    Construct a normal - lognormal mixture model.

    Parameters
    ----------
    t_data: ~np.array of float
        The input data to be fit.
    t_offp: ~np.array of float
        The off-pulse data.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()
    offp = t_offp.copy()

    # on-pulse mean and std
    onp_mean = np.mean(data)
    onp_std = np.std(data)
    print(f"On-pulse mean: {onp_mean:.5f}")
    print(f"On-pulse std: {onp_std:.5f}")

    # off-pulse mean and std
    offp_mean = np.mean(offp)
    offp_std = np.std(offp)
    print(f"Off-pulse mean: {offp_mean:.5f}")
    print(f"Off-pulse std: {offp_std:.5f}")

    coords = {"component": np.arange(3), "obs_id": np.arange(len(data))}

    with pm.Model(coords=coords) as model:
        x = pm.Data("x", data, dims="obs_id")

        # mixture weights
        w = pm.Dirichlet("w", a=np.array([0.3, 0.3, 0.7]), dims="component")

        # priors
        mu = pm.Normal(
            "mu",
            mu=np.array([offp_mean, 0.3, np.log(onp_mean)]),
            sigma=np.array([0.01, offp_std, np.log(1.75)]),
            dims="component",
        )
        sigma = pm.HalfNormal(
            "sigma",
            sigma=np.array([offp_std, offp_std, 1.0]),
            dims="component",
        )

        # 1) normal distribution for nulling
        # mu = location, sigma = scale
        norm1 = pm.Normal.dist(mu=mu[0], sigma=sigma[0])
        norm2 = pm.Normal.dist(mu=mu[1], sigma=sigma[1])

        # 2) lognormal distribution for pulses
        # mu = log of location, sigma = log of scale
        lognorm = pm.Lognormal.dist(mu=mu[2], sigma=sigma[2])

        components = [norm1, norm2, lognorm]

        pm.Mixture("obs", w=w, comp_dists=components, observed=x, dims="obs_id")

    return model


def normal_lognormal_analytic_pdf(x, w, mu, sigma, icomp):
    """
    Get the analytic PDF for the normal - lognormal model.

    Returns
    -------
    pdf: ~np.array of float
        The model PDF evaluated at the `x` values.
    """

    if icomp == 0:
        dist = pm.Normal.dist(mu=mu, sigma=sigma)
    elif icomp == 1:
        dist = pm.Normal.dist(mu=mu, sigma=sigma)
    elif icomp == 2:
        dist = pm.Lognormal.dist(mu=mu, sigma=sigma)
    else:
        raise NotImplementedError(f"Component not implemented: {icomp}")

    pdf = w[icomp] * pm.logp(dist, x).exp()

    return pdf.eval()


def normal_lognormal_mode(mu, sigma, icomp):
    """
    Compute the mode of the normal - lognormal model components.

    Returns
    -------
    mode: ~np.array of float
        The model PDF evaluated at the `x` values.
    """

    if icomp == 0:
        mode = mu
    elif icomp == 1:
        mode = pm.math.exp(mu - sigma**2)
    else:
        raise NotImplementedError(f"Component not implemented: {icomp}")

    return mode.eval()


def lognormal_lognormal(t_data, t_offp):
    """
    Construct a lognormal - lognormal mixture model.

    Parameters
    ----------
    t_data: ~np.array of float
        The input data to be fit.
    t_offp: ~np.array of float
        The off-pulse data.

    Returns
    -------
    model: ~pm.Model
        A mixture model.
    """

    data = t_data.copy()
    offp = t_offp.copy()

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
