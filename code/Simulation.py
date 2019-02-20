import copy
import numpy as np

from Observables import Observables

class Simulation:

    def __init__(self, system, parameters):
        self.initial_system = system
        self.parameters = parameters

    def vmc_cycle(self, initial_alpha, mc_iterations=10000, alpha_variation_iterations=10,
                  variation_step_width_alpha=0.025, use_importance_sampling=False):

        result = []

        for i in range(alpha_variation_iterations):
            alpha = initial_alpha + i * variation_step_width_alpha
            result.append(self.mc_cycle(alpha, use_importance_sampling, mc_iterations=mc_iterations))

        return result

    def mc_cycle(self, alpha, use_importance_sampling=False, mc_iterations=10000):

        # set up systems
        current_system = copy.deepcopy(self.initial_system)
        current_system.calculate_wave_function(alpha)
        trial_system = copy.deepcopy(current_system)

        # initialize averages
        avg = Observables()

        for i in range(mc_iterations):

            if use_importance_sampling:
                current_system.calculate_drift_force(alpha)

            current_system = self.mc_step(alpha, current_system, trial_system, use_importance_sampling)

            # calculate local energy of last configuration
            current_system.calculate_local_energy(alpha)
            current_system.calculate_wave_function(alpha)
            current_system.calculate_wave_function_derivative(alpha)

            # update averages
            avg.update_cumulative_quantities(current_system.local_energy, current_system.wave_function_value,
                                             current_system.wave_function_derivative)

        # finalize averages
        avg.finalize_averages(mc_iterations)

        return avg

    def mc_step(self, alpha, current_system, trial_system, use_importance_sampling=False):

        for j in range(current_system.particle_number):

            # generate trial configuration
            if use_importance_sampling:
                trial_system.particles[j].perturb_position_importance(current_system.drift_force[j, :])
                trial_system.calculate_drift_force(alpha)
            else:
                trial_system.particles[j].perturb_position_uniformly(self.parameters.update_radius)

            # calculate wave function of trial configuration
            trial_system.calculate_wave_function(alpha)

            # calculate acceptance probability
            acceptance_probability = trial_system.wave_function_value ** 2 / current_system.wave_function_value ** 2

            # multiply by greens function if using importance sampling
            if use_importance_sampling:
                acceptance_probability *= self.evaluate_greens_function(current_system.particles[j].position,
                                                                        trial_system.particles[j].position,
                                                                        current_system.drift_force[j, :],
                                                                        trial_system.drift_force[j, :])

            # update configuration or roll back changes in trial configuration
            if Simulation.check_acceptance(acceptance_probability):
                current_system.wave_function_value = trial_system.wave_function_value
                for k in range(len(self.initial_system.particles[j].position)):
                    current_system.particles[j].position[k] = trial_system.particles[j].position[k]
            else:
                for k in range(len(self.initial_system.particles[j].position)):
                    trial_system.particles[j].position[k] = current_system.particles[j].position[k]

        return current_system

    def gradient_descent(self, initial_alpha, tolerance=10**(-6), learning_rate=0.1,
                         mc_iterations=1000, max_iterations=25, use_importance_sampling=False):

        alpha = initial_alpha

        for i in range(max_iterations):

            # do one mc cycle
            result = self.mc_cycle(alpha, use_importance_sampling, mc_iterations=mc_iterations)

            # compute gradient
            local_energy_derivative = 2 * (result.wave_function_energy_average -
                                           result.wave_function_derivative_average * result.energy_average)

            alpha -= learning_rate * local_energy_derivative

            ## update learning rate or alpha
            #if (i > 0 and result.energy_average > old_energy):
            #    learning_rate /= 2.0
            #else:
            #    alpha -= learning_rate * local_energy_derivative

            # stop if convergence criterion satisfied
            if abs(sum(local_energy_derivative)) < tolerance:
                break

            # store previous energy
            old_energy = result.energy_average

        return alpha

    # --- static methods --- #

    @staticmethod
    def evaluate_greens_function(position_old, position_new, drift_force_old, drift_force_new,
                                 diffusion_coefficient=0.5, time_step=0.001):

        greens_function = 0
        for i in range(len(position_old)):
            greens_function += (position_old[i] - position_new[i]) * (drift_force_old[i] + drift_force_new[i]) + \
                               (0.5 * diffusion_coefficient * time_step) * \
                               (drift_force_old[i] ** 2 - drift_force_new[i] ** 2)

        #return np.exp(greens_function)
        return np.exp(0.5 * greens_function)

    @staticmethod
    def check_acceptance(acceptance_probability):

        if acceptance_probability >= 1 or np.random.rand() <= acceptance_probability:
            return True
        return False
