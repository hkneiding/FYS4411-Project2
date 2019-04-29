import numpy as np


class System:

    def __init__(self, particles, energy_model):
        self.particles = particles
        self.energy_model = energy_model

        self.wave_function_value = 0
        self.wave_function_derivative = 0
        self.local_energy = 0
        self.drift_force = np.zeros((len(self.particles), len(self.particles[0].position)))

        self.particle_number = len(self.particles)

    def calculate_wave_function(self):
        self.wave_function_value = self.energy_model.wave_function.evaluate(self.get_particle_position_array().flatten())

    def calculate_local_energy(self):
        self.local_energy = self.energy_model.calculate_local_energy(self.get_particle_position_array())

    def calculate_drift_force(self):
        self.drift_force = self.energy_model.calculate_drift_force(self.get_particle_position_array())

    def calculate_wave_function_derivative(self):
        self.wave_function_derivative = self.energy_model.wave_function.calculate_derivative(
            self.get_particle_position_array().flatten())

    def get_particle_position_array(self):

        x = []
        for i in range(len(self.particles)):
            x.append(self.particles[i].position)
        y = np.array(x)

        return y
