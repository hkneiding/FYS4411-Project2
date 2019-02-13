import numpy as np

from numba import jitclass
from numba import float64, float32, int32, int64

specs = [
    ('particle_positions', float64[:, :]),
]


# cannot deep copy jitted classes
#@jitclass(specs)
class Gaussian:

    def __init__(self, particle_positions):
        self.particle_positions = particle_positions

    def evaluate(self, alpha):

        gaussian = 0
        for i in range(len(self.particle_positions)):
            r2 = 0
            for j in range(len(self.particle_positions[0,:])):
                r2 += self.particle_positions[i,j] ** 2
            gaussian += r2

        return np.exp(- alpha * gaussian)

    def calculate_gradient(self, alpha):

        gradient = np.zeros((len(self.particle_positions), len(self.particle_positions[0, :])))
        for i in range(len(self.particle_positions)):
            for j in range(len(self.particle_positions[0, :])):
                gradient[i,j] = - 2 * alpha * self.particle_positions[i,j]

        return gradient

    def calculate_laplacian(self, alpha):

        laplacian = 0
        for i in range(len(self.particle_positions)):
            r2 = 0
            for j in range(len(self.particle_positions[0,:])):
                r2 += self.particle_positions[i, j] ** 2
            laplacian += 2 * alpha * r2 - len(self.particle_positions[0,:])

        return 2 * alpha * laplacian

    def calculate_laplacian_numerically(self, alpha, step_size=0.001):

        squared_step_size = step_size ** 2
        laplacian = 0
        for i in range(len(self.particle_positions)):
            for j in range(len(self.particle_positions[0,:])):

                self.particle_positions[i, j] += step_size
                positive_step = self.evaluate(alpha)

                self.particle_positions[i, j] -= 2 * step_size
                negative_step = self.evaluate(alpha)

                self.particle_positions[i, j] += step_size
                middle_step = self.evaluate(alpha)

                laplacian += (positive_step - 2 * middle_step + negative_step) / (middle_step * squared_step_size)

        return laplacian
