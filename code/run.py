import time
import numpy as np

from Observables import Observables
from System import System
from Particle import Particle
from Simulation import Simulation
from NeuralQuantumState import NeuralQuantumState
from Hamiltonian import Hamiltonian

"THIS AREA"

particles = 1
dimensions = 1

use_importance_sampling=True
learning_rate=0.05
update_radius=0.45
time_step=0.001
max_iterations=100
tolerance=10**(-6)
mc_iterations = 50000






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

p = generate_particles(particles, dimensions)

nqs = NeuralQuantumState(initial_sigma=1)
h = Hamiltonian(nqs)

sys = System(p, h)

sim = Simulation(sys)

nqs_opt = sim.stochastic_gradient_descent(tolerance, learning_rate, mc_iterations, max_iterations, use_importance_sampling, update_radius, time_step)


print(sim.mc_cycle().energy_average)