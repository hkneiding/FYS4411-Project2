import unittest
import numpy as np
import numpy.testing as npt

from System import System
from Particle import Particle
from Simulation import Simulation
from Parameters import Parameters
from hamiltonians.HarmonicOscillator import HarmonicOscillator
from wavefunctions.Gaussian import Gaussian


class test_Simulation(unittest.TestCase):

    def test_mc_cycle(self):

        wf = Gaussian()
        energy_model = HarmonicOscillator(wf, 1)
        d = np.random.randint(1,4)
        N = np.random.randint(2,10)
        particles = generate_particles(N,d)
        system = System(particles, energy_model)
        parameters = Parameters(1)
        sim = Simulation(system, parameters)
        alpha = np.array([0.5])
        result = sim.mc_cycle(alpha, mc_iterations=1000, use_importance_sampling=False)

        npt.assert_almost_equal(result.energy_average, 0.5 * d * N)

    def test_mc_cycle_importance_sampling(self):

        wf = Gaussian()
        energy_model = HarmonicOscillator(wf, 1)
        d = np.random.randint(1,4)
        N = np.random.randint(2,10)
        particles = generate_particles(N,d)
        system = System(particles, energy_model)
        parameters = Parameters(1)
        sim = Simulation(system, parameters)
        alpha = np.array([0.5])
        result = sim.mc_cycle(alpha, mc_iterations=1000, use_importance_sampling=True)

        npt.assert_almost_equal(result.energy_average, 0.5 * d * N)

    def test_gradient_descent(self):

        wf = Gaussian()
        energy_model = HarmonicOscillator(wf, 1)
        d = np.random.randint(1,4)
        N = np.random.randint(2,10)
        particles = generate_particles(N,d)
        system = System(particles, energy_model)
        parameters = Parameters(1)
        sim = Simulation(system, parameters)
        initial_alpha = np.array([0.75])
        result = sim.gradient_descent(initial_alpha, max_iterations=100, mc_iterations=1000,
                                      use_importance_sampling=False)

        npt.assert_almost_equal(result, 0.5)

    def test_gradient_descent_importance_sampling(self):

        wf = Gaussian()
        energy_model = HarmonicOscillator(wf, 1)
        d = np.random.randint(1,4)
        N = np.random.randint(2,10)
        particles = generate_particles(N,d)
        system = System(particles, energy_model)
        parameters = Parameters(1)
        sim = Simulation(system, parameters)
        initial_alpha = np.array([0.75])
        result = sim.gradient_descent(initial_alpha, max_iterations=100, mc_iterations=1000,
                                      use_importance_sampling=True)

        npt.assert_almost_equal(result, 0.5)

def generate_particle(d):

    r = np.zeros(d)
    for i in range(d):
        r[i] = np.random.rand() - 0.5
    return Particle(r)

def generate_particles(N, d):

    particles = []
    for i in range(N):
        particles.append(generate_particle(d))
    return particles