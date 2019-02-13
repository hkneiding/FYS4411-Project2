class HarmonicOscillator:

    def __init__(self, wave_function, omega, use_numerical_differentiation=False):
        self.wave_function = wave_function
        self.omega = omega
        self.use_numerical_differentiation = use_numerical_differentiation

    def calculate_local_energy(self, particle_positions, alpha):
        return self.calculate_potential_energy(particle_positions) + \
               self.calculate_kinetic_energy(alpha, use_numerical_differentiation=self.use_numerical_differentiation)

    def calculate_potential_energy(self, particle_positions):

        potential_energy = 0
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0, :])):
                r2 += particle_positions[i,j] ** 2

            potential_energy += r2
        return potential_energy * 0.5 * self.omega ** 2

    def calculate_kinetic_energy(self, alpha, use_numerical_differentiation=False):

        if use_numerical_differentiation:
            kinetic_energy = - 0.5 * self.wave_function.calculate_laplacian_numerically(alpha)
        else:
            kinetic_energy = - 0.5 * self.wave_function.calculate_laplacian(alpha)
        return kinetic_energy

    def calculate_drift_force(self, alpha):

        return 2 * self.wave_function.calculate_gradient(alpha)
