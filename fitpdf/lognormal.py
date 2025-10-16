#
#   2025 Fabian Jankowski
#   Lognormal mixture models.
#

import logging

import numpy as np
import pymc as pm

from fitpdf.models import Model


class Lognormal(Model):
    name = "Lognormal - lognormal"

    def __init__(self):
        """
        Model distribution.
        """

        self.__log = logging.getLogger("fitpdf.models")

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

    def get_analytic_pdf(self, x, w, mu, sigma, icomp):
        """
        Get the analytic PDF.

        Returns
        -------
        pdf: ~np.array of float
            The model PDF evaluated at the `x` values.
        """
        pass

    def get_mode(self, mu, sigma, icomp):
        """
        Compute the mode of the model component.

        Returns
        -------
        mode: ~np.array of float
            The mode of the model component.
        """
        pass
