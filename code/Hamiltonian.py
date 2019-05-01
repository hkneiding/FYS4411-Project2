import numpy as np


class Hamiltonian:

    def __init__(self, wave_function, include_interaction=False):

        self.omega = 1
        self.include_interaction = include_interaction
        self.wave_function = wave_function

    def calculate_local_energy(self, particle_positions):

        local_energy = self.calculate_potential_energy(particle_positions) + \
                       self.calculate_kinetic_energy(particle_positions)

        if self.include_interaction:
            local_energy += self.calculate_interaction_contribution(particle_positions)

        return local_energy

    def calculate_potential_energy(self, particle_positions):

        potential_energy = 0
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0, :])):
                r2 += particle_positions[i, j] ** 2
            potential_energy += r2
        return potential_energy * 0.5 * self.omega ** 2

    def calculate_kinetic_energy(self, particle_positions):

        kinetic_energy = 0.5 * self.wave_function.calculate_laplacian(particle_positions.flatten())
        return kinetic_energy

    def calculate_interaction_contribution(self, particle_positions):
        print("int")
        interaction_contribution = 0
        for i in range(len(particle_positions)):
            for j in range(i + 1, len(particle_positions)):
                interaction_contribution += np.linalg.norm(particle_positions[i, :] - particle_positions[j, :])

        return interaction_contribution

    def calculate_drift_force(self, particle_positions):

        drift_force = np.zeros(len(particle_positions.flatten()))
        for i in range(len(drift_force)):
            drift_force[i] = 2 * self.wave_function.calculate_gradient(particle_positions.flatten(), i)

        return drift_force.reshape((len(particle_positions), len(particle_positions[0, :])))
