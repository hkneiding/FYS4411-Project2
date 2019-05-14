import copy
import numpy as np
import sys

from Observables import Observables


class Simulation:

    def __init__(self, system):
        self.system = system

    def mc_cycle(self, mc_iterations=10000, sampling="mc", update_radius=1, time_step=0.001, burn_in_percentage=0.1):

        # set up systems
        current_system = copy.deepcopy(self.system)
        current_system.calculate_wave_function()

        # initialize averages
        avg = Observables((1 - burn_in_percentage) * mc_iterations)

        for i in range(mc_iterations):

            if sampling == "importance": #use_importance_sampling:
                current_system.calculate_drift_force()

            current_system, acceptance_rate = self.mc_step(current_system, copy.deepcopy(current_system), sampling, update_radius=update_radius, time_step=time_step)

            # calculate local energy of last configuration
            current_system.calculate_local_energy()
            current_system.calculate_wave_function()
            current_system.calculate_wave_function_derivative()

            #if i > burn_in_percentage * mc_iterations:

            # update averages
            avg.acceptance_rate += acceptance_rate
            avg.update_cumulative_quantities(current_system.local_energy, current_system.wave_function_value,
                                                 current_system.wave_function_derivative)

            avg.positions.append(current_system.particles[0].position.tolist())

        # finalize averages
        avg.finalize_averages()
        

        # save final configuration
        self.system = current_system

        return avg

    def mc_step(self, current_system, trial_system, sampling="mc", update_radius=1, time_step=0.001):
        """ I tried cutting down the number of if/else tests to speed things up a bit (?).
        I also changed from use_importance-sampling = True/False to sampling= type"""
        accepted_steps = 0

        # randomly select a particle
        j = np.random.randint(low=0, high=current_system.particle_number)

        # generate trial configuration
        if sampling == "mc":
            trial_system.particles[j].perturb_position_uniformly(update_radius)
            # calculate wave function of trial configuration
            trial_system.calculate_wave_function()
            # calculate acceptance probability 
            acceptance_probability = trial_system.wave_function_value ** 2 / current_system.wave_function_value ** 2
        
        elif sampling == "importance":
            trial_system.particles[j].perturb_position_importance(current_system.drift_force[j, :], time_step=time_step)
            trial_system.calculate_drift_force()
            # calculate trial wave function
            trial_system.calculate_wave_function()
            # calculate acceptance probablilty
            acceptance_probability = trial_system.wave_function_value ** 2 / current_system.wave_function_value ** 2
            acceptance_probability *= self.evaluate_greens_function(current_system.particles[j].position, trial_system.particles[j].position,current_system.drift_force[j, :], trial_system.drift_force[j, :],time_step=time_step)

        elif sampling == "gibbs":
            #  Here, gibbs will be implemented, but I don't want to update too much at once :)
            acceptance_probability = 1
            print("yessda")

        else:
            sys.exit("error, sampling is wrong.")


        # update configuration or roll back changes in trial configuration
        if Simulation.check_acceptance(acceptance_probability):
            accepted_steps += 1
            current_system.wave_function_value = trial_system.wave_function_value
            for k in range(len(self.system.particles[j].position)):
                current_system.particles[j].position[k] = trial_system.particles[j].position[k]
        else:
            for k in range(len(self.system.particles[j].position)):
                trial_system.particles[j].position[k] = current_system.particles[j].position[k]

        return current_system, accepted_steps

    def stochastic_gradient_descent(self, tolerance=10**(-6), learning_rate_init=0.01, mc_iterations=50000, max_iterations=25, sampling="mc", update_radius=1, time_step=0.001, burn_in_percentage=0.1, no_iteration=111):
        #A = []
        #B = []
        #W = []
        E = np.zeros(max_iterations)

        for i in range(max_iterations):
            learning_rate = (1 - 0.9*i/max_iterations) * learning_rate_init

            #A.append(self.system.energy_model.wave_function.a)
            #B.append(self.system.energy_model.wave_function.b)
            #W.append(self.system.energy_model.wave_function.w)


            result = self.mc_cycle(sampling=sampling, mc_iterations=mc_iterations, update_radius=update_radius, time_step=time_step, burn_in_percentage=burn_in_percentage)
            E[i] = result.energy_average
            # compute gradient
            local_energy_derivative = 2 * (result.wave_function_energy_average - result.wave_function_derivative_average * result.energy_average)

            for j in range(self.system.energy_model.wave_function.nx):
                self.system.energy_model.wave_function.a[j] -=  learning_rate * local_energy_derivative[j]

            for j in range(self.system.energy_model.wave_function.nh):
                self.system.energy_model.wave_function.b[j] -= \
                    learning_rate * local_energy_derivative[self.system.energy_model.wave_function.nx + j]

            k = self.system.energy_model.wave_function.nx + self.system.energy_model.wave_function.nh

            for j in range(self.system.energy_model.wave_function.nx):
                for l in range(self.system.energy_model.wave_function.nh):
                    self.system.energy_model.wave_function.w[j, l] -= learning_rate * local_energy_derivative[k]
                    k += 1

            #print(sum(abs(local_energy_derivative)))
            #print("Energy: " + str(result.energy_average))

            # stop if convergence criterion satisfied
            if abs(sum(local_energy_derivative)) < tolerance:
                break
        
        #np.save("results/A%03d.npy" % no_iteration, np.array(A))
        #np.save("results/B%03d.npy" % no_iteration, np.array(B))
        #np.save("results/W%03d.npy" % no_iteration, np.array(W))
        np.save("results/E%03d.npy" % no_iteration, E)

        return self.system.energy_model.wave_function

    # --- static methods --- #

    @staticmethod
    def evaluate_greens_function(position_old, position_new, drift_force_old, drift_force_new,
                                 diffusion_coefficient=0.5, time_step=0.001):

        greens_function = 0
        for i in range(len(position_old)):

            greens_function += (position_old[i] - position_new[i]) * (drift_force_old[i] + drift_force_new[i]) + \
                               (0.5 * diffusion_coefficient * time_step) * \
                               (drift_force_old[i] ** 2 - drift_force_new[i] ** 2)

        return np.exp(0.5 * greens_function)

    @staticmethod
    def check_acceptance(acceptance_probability):

        if acceptance_probability >= 1 or np.random.rand() <= acceptance_probability:
            return True
        return False
