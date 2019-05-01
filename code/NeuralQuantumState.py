import numpy as np


class NeuralQuantumState:

    def __init__(self, initial_sigma, nx, nh, dim):

        self.nx = nx
        self.nh = nh
        self.dim = dim
        self.sigma_squared = initial_sigma ** 2

        self.h = np.zeros(self.nh)
        self.a = np.zeros(self.nx)
        self.b = np.zeros(self.nh)
        self.w = np.zeros((self.nx, self.nh))

        self.prefactor = 1
        #self.prefactor = 0.5

        self.initialize_weights(initial_sigma)

    def initialize_weights(self, initial_sigma):

        self.a = np.random.normal(0, initial_sigma, self.nx)
        self.b = np.random.normal(0, initial_sigma, self.nh)
        self.w = np.random.normal(0, initial_sigma, (self.nx, self.nh))

    def evaluate(self, x):

        term_1 = (x - self.a).dot(x - self.a)
        term_1 = np.exp(- term_1 / (2 * self.sigma_squared))

        term_2 = 1
        Q = self.calculate_Q(x)
        for i in range(self.nh):
            term_2 *= (1 + np.exp(Q[i]))

        return term_1 * term_2

    def calculate_laplacian(self, x):

        laplacian = 0

        sigmoid_Q = self.calculate_sigmoid_Q(x)
        derivative_sigmoid_Q = self.calculate_sigmoid_derivative_Q(x)

        for i in range(self.nx):
            d1_ln_psi = - (x[i] - self.a[i]) / self.sigma_squared + self.w[i, :].dot(sigmoid_Q) / self.sigma_squared

            summation = 0

            for j in range(self.nh):
                summation += derivative_sigmoid_Q[j] * self.w[i, j] ** 2

            d2_ln_psi = - 1.0 / self.sigma_squared + summation / (self.sigma_squared ** 2)

            d1_ln_psi *= self.prefactor
            d2_ln_psi *= self.prefactor

            laplacian += - d1_ln_psi * d1_ln_psi - d2_ln_psi

        return laplacian

    def calculate_gradient(self, x, index):

        summation = self.calculate_sigmoid_Q(x).dot(self.w[index, :])
        return self.prefactor * (-(x[index] - self.a[index]) / self.sigma_squared +
                                 summation / self.sigma_squared)

    def calculate_derivative(self, x):

        sigmoid_Q = self.calculate_sigmoid_Q(x)

        d_psi = np.zeros(self.nx + self.nh + self.nx * self.nh)

        for i in range(self.nx):
            d_psi[i] = (x[i] - self.a[i]) / self.sigma_squared

        for i in range(self.nx, self.nx + self.nh):
            d_psi[i] = sigmoid_Q[i - self.nx]

        k = self.nx + self.nh
        for i in range(self.nx):
            for j in range(self.nh):
                d_psi[k] = x[i] * sigmoid_Q[j] / self.sigma_squared
                k += 1

        return self.prefactor * d_psi

    def calculate_Q(self, x):

        return self.b + (((1.0 / self.sigma_squared) * x).transpose().dot(self.w).transpose())

    def calculate_sigmoid_Q(self, x):

        return 1 / (1 + np.exp(- self.calculate_Q(x)))

    def calculate_sigmoid_derivative_Q(self, x):

        exp_Q = np.exp(self.calculate_Q(x))
        derivative_sigmoid_Q = exp_Q / (1 + exp_Q) ** 2

        return derivative_sigmoid_Q
