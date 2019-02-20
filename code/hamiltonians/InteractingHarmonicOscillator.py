import numpy as np


class InteractingHarmonicOscillator:

    def __init__(self, wave_function, gamma, a, use_numerical_differentiation=False):
        self.wave_function = wave_function
        self.gamma = gamma
        self.a = a
        self.use_numerical_differentiation = use_numerical_differentiation

    def calculate_local_energy(self, particle_positions, alpha):
        return self.calculate_potential_energy(particle_positions) + \
               self.calculate_kinetic_energy(particle_positions, alpha,
                                             use_numerical_differentiation=self.use_numerical_differentiation)

    def calculate_potential_energy(self, particle_positions):

        potential_energy = 0
        for i in range(len(particle_positions)):

            # single particle contribution
            r2 = 0
            for j in range(len(particle_positions[0, :])):
                if j == 2:
                    r2 += (self.gamma ** 2) * (particle_positions[i, j] ** 2)
                else:
                    r2 += particle_positions[i, j] ** 2
            potential_energy += r2

            # interaction contribution
            for j in range(i+1, len(particle_positions)):
                if np.linalg.norm(particle_positions[i,:] - particle_positions[j,:]) <= self.a:
                    potential_energy += 1000000  # ???

        return 0.5 * potential_energy

    def calculate_kinetic_energy(self, particle_positions, alpha, use_numerical_differentiation=False):

        if use_numerical_differentiation:
            kinetic_energy = - 0.5 * self.wave_function.calculate_laplacian_numerically(particle_positions, alpha)
        else:
            kinetic_energy = - 0.5 * self.wave_function.calculate_laplacian(particle_positions, alpha)
        return kinetic_energy

    def calculate_drift_force(self, particle_positions, alpha):

        return 2 * self.wave_function.calculate_gradient(particle_positions, alpha)
