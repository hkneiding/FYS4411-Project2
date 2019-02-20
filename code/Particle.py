import numpy as np


class Particle:

    def __init__(self, position):
        if not isinstance(position, np.ndarray):
            raise TypeError('position have to be an array')
        self.position = position

    def perturb_position_uniformly(self, update_radius):
        d = len(self.position)
        for i in range(0, d):
            self.position[i] += update_radius * np.random.uniform(-0.5, 0.5)

    def perturb_position_importance(self, drift_force, diffusion_coefficient=0.5, time_step=0.001):
        d = len(self.position)
        for i in range(0, d):
            self.position[i] += drift_force[i] * time_step * diffusion_coefficient +\
                                np.sqrt(time_step) * np.random.normal()
