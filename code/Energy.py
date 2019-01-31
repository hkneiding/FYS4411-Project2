import numpy as np


class Energy:

    def __init__(self):
        self.local_energy = 0
        self.one_body_potential = 0
        self.two_body_potential = 0

    def calculate_local_energy(self):
        self.local_energy = self.one_body_potential + self.two_body_potential
