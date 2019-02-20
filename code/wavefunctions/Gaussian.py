import numpy as np


class Gaussian:

    def __init__(self):
        return

    def evaluate(self, particle_positions, alpha):

        gaussian = 0
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0,:])):
                r2 += particle_positions[i,j] ** 2
            gaussian += r2

        return np.exp(- alpha[0] * gaussian)

    def calculate_gradient(self, particle_positions, alpha):

        gradient = np.zeros((len(particle_positions), len(particle_positions[0, :])))
        for i in range(len(particle_positions)):
            for j in range(len(particle_positions[0, :])):
                gradient[i,j] = - 2 * alpha[0] * particle_positions[i,j]

        return gradient

    def calculate_laplacian(self, particle_positions, alpha):

        laplacian = 0
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0,:])):
                r2 += particle_positions[i, j] ** 2
            laplacian += 2 * alpha[0] * r2 - len(particle_positions[0,:])

        return 2 * alpha[0] * laplacian

    def calculate_laplacian_numerically(self, particle_positions, alpha, step_size=0.001):

        squared_step_size = step_size ** 2
        laplacian = 0
        for i in range(len(particle_positions)):
            for j in range(len(particle_positions[0,:])):

                particle_positions[i, j] += step_size
                positive_step = self.evaluate(particle_positions, alpha)

                particle_positions[i, j] -= 2 * step_size
                negative_step = self.evaluate(particle_positions, alpha)

                particle_positions[i, j] += step_size
                middle_step = self.evaluate(particle_positions, alpha)

                laplacian += (positive_step - 2 * middle_step + negative_step) / (middle_step * squared_step_size)

        return laplacian

    def calculate_derivative(self, particle_positions, alpha):

        r2_sum = 0
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0, :])):
                r2 += particle_positions[i, j] ** 2
            r2_sum += r2

        return np.array([- r2_sum * np.exp(- alpha[0] * r2_sum)])
