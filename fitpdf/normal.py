#
#   2025 Fabian Jankowski
#   Normal mixture models.
#

import logging

import numpy as np
import pymc as pm

from fitpdf.model import Model


class Normal(Model):
    name = "Normal - normal"

    def __init__(self):
        """
        Model distribution.
        """

        self.__log = logging.getLogger("fitpdf.models")

        self.ncomp = 2

    def __repr__(self):
        """
        Representation of the object.
        """

        info_dict = {"bla": "XXX"}

        info_str = "{0}".format(info_dict)

        return info_str

    def __str__(self):
        """
        String representation of the object.
        """

        info_str = "{0}: {1}".format(self.name, repr(self))

        return info_str

    def get_model(self, t_data, t_offp):
        """
        Construct a mixture model.

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

        coords = {"component": np.arange(2), "obs_id": np.arange(len(data))}

        with pm.Model(coords=coords) as model:
            x = pm.Data("x", data, dims="obs_id")

            # mixture weights
            w = pm.Dirichlet("w", a=np.array([1.0, 1.0]), dims="component")

            # priors
            mu = pm.Normal("mu", mu=np.array([0.0, 1.0]), sigma=1, dims="component")
            sigma = pm.HalfNormal("sigma", sigma=np.array([1.0, 1.0]), dims="component")

            # 1) normal distribution for nulling
            # 2) normal distribution for pulses
            components = pm.Normal.dist(
                mu=mu,
                sigma=sigma,
                shape=(2,),
            )

            pm.Mixture("obs", w=w, comp_dists=components, observed=x, dims="obs_id")

        return model

    def get_analytic_pdf(self, x, w, mu, sigma, icomp):
        """
        Get the analytic PDF.

        Returns
        -------
        pdf: ~np.array of float
            The model PDF evaluated at the `x` values.
        """

        if icomp == 0:
            dist = pm.Normal.dist(mu=mu, sigma=sigma)
        elif icomp == 1:
            dist = pm.Normal.dist(mu=mu, sigma=sigma)
        else:
            raise NotImplementedError(f"Component not implemented: {icomp}")

        pdf = w[icomp] * pm.logp(dist, x).exp()

        return pdf.eval()

    def get_mode(self, mu, sigma, icomp):
        """
        Compute the mode of the model component.

        Returns
        -------
        modes: ~np.array of float
            The model PDF evaluated at the `x` values.
        """

        if icomp in [0, 1]:
            mode = mu
        else:
            raise NotImplementedError(f"Component not implemented: {icomp}")

        return mode.eval()
