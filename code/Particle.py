import numpy as np


class Particle:

    def __init__(self, position):
        if not isinstance(position, np.ndarray):
            raise TypeError('position have to be an array')
        self.position = position
