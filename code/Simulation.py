import numpy as np

from MonteCarlo import MonteCarlo


class Simulation:

    def __init__(self, system, parameters):
        self.initial_system = system
        self.parameters = parameters

    def vmc_cycle(self, initial_alpha, initial_beta, a, mc_iterations=10000, variation_iterations=10,
                  variation_step_width_alpha=0.025, variation_step_width_beta=0.01):

        result = np.zeros((variation_iterations * variation_iterations, 6))

        for i in range(variation_iterations):
            alpha = initial_alpha + i * variation_step_width_alpha
            for j in range(variation_iterations):
                beta = initial_beta + j * variation_step_width_beta

                k = i * variation_iterations + j
                result[k,0] = alpha
                result[k,1] = beta
                result[k,2], result[k,3], result[k,4], result[k,5] = self.mc_cycle(alpha, beta, a, mc_iterations)

        return result

    def mc_cycle(self, alpha, beta, a, mc_iterations=10000):

        old_system = self.initial_system
        old_system.calculate_wave_function(alpha, beta, a)

        energy_average = 0
        squared_energy_average = 0

        for i in range(mc_iterations):

            # generate trial configuration
            trial_system = MonteCarlo.generate_trial_configuration(old_system, self.parameters)
            # calculate wave function of trial configuration
            trial_system.calculate_wave_function(alpha, beta, a)
            # append accepted configuration to trajectory
            old_system = MonteCarlo.evaluate_trial_configuration_greedy(old_system, trial_system)
            # calculate local energy of last configuration
            old_system.calculate_energy()

            # update averages
            energy_average += old_system.energy.local_energy
            squared_energy_average += old_system.energy.local_energy ** 2

        energy_average /= mc_iterations
        squared_energy_average /= mc_iterations
        variance = squared_energy_average - energy_average ** 2
        error = np.sqrt(variance / mc_iterations)

        return energy_average, squared_energy_average, variance, error
