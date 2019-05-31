import numpy as np


class Observables:

    def __init__(self, N):
        self.cumulative_energy = 0
        self.cumulative_squared_energy = 0
        self.cumulative_wave_function_derivative = 0
        self.cumulative_wave_function_energy = 0

        self.energy_average = 0
        self.energy_squared_average = 0
        self.wave_function_derivative_average = 0
        self.wave_function_energy_average = 0

        self.energy_trajectory = []
        self.positions = []

        self.variance = 0
        self.error = 0

        self.acceptance_rate = 0
        self.N = N

    def update_cumulative_quantities(self, local_energy, wave_function, wave_function_derivative):
        """ Updating averages within the mc cycle """

        self.cumulative_energy += local_energy
        self.cumulative_squared_energy += local_energy ** 2
        self.cumulative_wave_function_derivative += wave_function_derivative #/ wave_function
        self.cumulative_wave_function_energy += wave_function_derivative * local_energy #/ wave_function
        self.energy_trajectory.append(local_energy)
    

    def finalize_averages(self):
        """ Finalize all averages and errors from run """
        self.compute_averages()
        self.compute_error()
        self.acceptance_rate = self.acceptance_rate / self.N
        #self.print_averages()

    def compute_averages(self):
        """ Computing final averages for finalize averages """
        self.energy_average = self.cumulative_energy / self.N
        self.energy_squared_average = self.cumulative_squared_energy / self.N
        self.wave_function_derivative_average = self.cumulative_wave_function_derivative / self.N
        self.wave_function_energy_average = self.cumulative_wave_function_energy / self.N

    def compute_error(self):
        """ Computing final errors for the finalize_averages """
        self.variance = self.energy_squared_average - self.energy_average ** 2
        self.error = np.sqrt(abs(self.variance) / self.N)
    
    def print_averages(self):
        print("-------------------------------------")
        print("<E>               : ", self.energy_average)
        print("<E^2>             : ", self.energy_squared_average)
        print("Variance          : ", self.variance)
        print("Error, STD        : ", self.error)
        print("Acceptance rate   : ", self.acceptance_rate)
        print("-------------------------------------\n")
