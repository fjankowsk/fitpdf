#
#   2025 Fabian Jankowski
#   Distribution model template.
#

import logging


class Model(object):
    name = "model"

    def __init__(self):
        """
        Model distribution.
        """

        self.__log = logging.getLogger("fitpdf.models")

        self.ncomp = None

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
        pass

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
