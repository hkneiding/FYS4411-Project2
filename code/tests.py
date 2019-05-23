import time
import numpy as np
import sys



from Observables import Observables
from System import System
from Particle import Particle
from Simulation import Simulation
from NeuralQuantumState import NeuralQuantumState
from Hamiltonian import Hamiltonian


#start_time = time.time()

particles = 1
dimensions = 1

sampling="mc"

max_iterations = 500
mc_iterations = 100

tolerance=0#10**(-9)
#learning_rate=0.01
update_radius=0.45
#time_step=0.45
"""max_iterations=1500
tolerance=10**(-6)
mc_iterations = 250"""

burn_in_percentage = 0.

# Neural Quantum state input:
initial_sigma=1
nx = 1
nh = 2



def generate_particle(d):
    r = np.zeros(d)
    for i in range(d):
        r[i] = np.random.rand() - 0.5
    return Particle(r)


def generate_particles(N, d):
    particle_positions = []
    for i in range(N):
        particle_positions.append(generate_particle(d))
    return particle_positions

#no_iteration = 0
#no_iteration_array = []


def run_tests(learning_rate, time_step, no_iteration):

    #no_iteration = int(np.load("results/counter.npy"))
    #np.save("counter.npy", np.array([1 + no_iteration]))
    print("Iteration began:  ", no_iteration)
    meta = open("results/iteration_metadata.txt", "a+")
    meta.write(str(no_iteration) +" " + str(learning_rate) +" "+ str(time_step)+ "\n")
    meta.close()


    p = generate_particles(particles, dimensions)

    nqs = NeuralQuantumState(initial_sigma, nx, nh, dimensions)
    h = Hamiltonian(nqs)

    sys = System(p, h)

    sim = Simulation(sys)

    nqs_opt = sim.stochastic_gradient_descent(tolerance, learning_rate, mc_iterations, max_iterations, sampling, update_radius, time_step, burn_in_percentage, no_iteration)

    #end_time = time.time()
    #print("Time spent: ", end_time - start_time, " seconds.")
    return 5*learning_rate

if __name__ == "__main__":
    run_tests(learning_rate=1.0, time_step=0.94, no_iteration=0)


"""
for learning_rate in np.linspace(0.001,0.3, 6):
    for time_step in np.linspace(0.01, 0.9, 6):
        print("iteration: ", no_iteration)
        p = generate_particles(particles, dimensions)

        nqs = NeuralQuantumState(initial_sigma, nx, nh, dimensions)
        h = Hamiltonian(nqs)

        sys = System(p, h)

        sim = Simulation(sys)

        nqs_opt = sim.stochastic_gradient_descent(tolerance, learning_rate, mc_iterations, max_iterations, use_importance_sampling, update_radius, time_step, burn_in_percentage, no_iteration)

        end_time = time.time()
        print("Time spent: ", end_time - start_time, " seconds.")
        no_iteration_array.append(np.array([learning_rate, time_step]))
        no_iteration += 1
"""
#np.save("iteation_parameters.npy", #np.array(no_iteration_array))

#print(sim.mc_cycle().energy_average)