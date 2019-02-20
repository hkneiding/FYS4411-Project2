import unittest
import numpy as np
import numpy.testing as npt

from wavefunctions.Gaussian import Gaussian


class test_Gaussian(unittest.TestCase):

    def test_caclculate_laplacian(self):

        alpha = np.array([0.5, 0.5])
        pos = np.random.rand(3,3) - 0.5
        wf = Gaussian()
        analytical_result = wf.calculate_laplacian(pos, alpha)
        numerical_result = wf.calculate_laplacian_numerically(pos, alpha)

        npt.assert_almost_equal(analytical_result, numerical_result, decimal=4)
