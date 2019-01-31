import numpy as np

from Energy import Energy


class System:

    def __init__(self, particles):
        self.particles = particles
        self.wave_function = 0
        self.energy = Energy()

    def calculate_wave_function(self, alpha, beta, a):

        wf = 1

        for i in range(len(self.particles)):
            one_body_part = 0
            if len(self.particles[0].position) == 1:
                one_body_part = np.exp(- alpha * (self.particles[i].position[0] ** 2))
            elif len(self.particles[0].position) == 2:
                one_body_part = np.exp(- alpha * (self.particles[i].position[0] ** 2
                                                  + self.particles[i].position[1] ** 2))
            elif len(self.particles[0].position) == 3:
                one_body_part = np.exp(- alpha * (self.particles[i].position[0] ** 2
                                                  + self.particles[i].position[1] ** 2
                                                  + beta * self.particles[i].position[2] ** 2))

            for j in range(i + 1, len(self.particles)):
                two_body_part = 1
                distance = np.linalg.norm(self.particles[i].position - self.particles[j].position)
                if distance <= a:
                    two_body_part *= 0
                else:
                    two_body_part *= 1 - (a / distance)

                wf *= one_body_part * two_body_part

        self.wave_function = wf

    def calculate_energy(self):
        self.energy.local_energy = 0

    def get_particle_position_array(self):

        x = []
        for i in range(len(self.particles)):
            x.append(self.particles[i].position)
        y = np.array(x)

        return y
