import numpy as np


class InteractingGaussian:

    def __init__(self, a):
        self.a = a
        return

    def evaluate(self, particle_positions, alpha):

        r2_sum = 0
        correlation = 1
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0,:])):
                if j == 2:
                    r2 += alpha[1] * (particle_positions[i, j] ** 2)
                else:
                    r2 += particle_positions[i, j] ** 2
            r2_sum += r2

            # interaction contribution
            for j in range(i+1, len(particle_positions)):
                r_ij = np.linalg.norm(particle_positions[i,:] - particle_positions[j,:])
                if r_ij <= self.a:
                    correlation *= 0
                    return 0
                else:
                    correlation *= 1 - (self.a / r_ij)

        return np.exp(- alpha[0] * r2_sum) * correlation

    def calculate_gradient(self, particle_positions, alpha):

        return

    def calculate_laplacian(self, particle_positions, alpha):

        term_1 = 0
        term_2 = 0
        term_3 = 0
        term_4 = 0

        for i in range(len(particle_positions)):

            # shortcuts for particle i positions
            xi = particle_positions[i, 0]
            yi = particle_positions[i, 1]
            zi = particle_positions[i, 2]

            # i vector
            ri = np.array([xi, yi, zi])
            ri2 = (xi ** 2) + (yi ** 2) + (alpha[1] * zi) ** 2

            # update term 1
            term_1 += 2 * alpha[0] * (2 * alpha[0] * ri2 - 2 - alpha[1])

            for j in range(len(particle_positions)):

                if i == j:
                    continue
                else:

                    # j vector
                    rj = np.array([particle_positions[j, 0],
                                   particle_positions[j, 1],
                                   particle_positions[j, 2]])

                    # calculate Euclidean distance between particle i and j
                    dij = np.linalg.norm(particle_positions[i,:] - particle_positions[j,:])

                    # calculate u first derivative (distance between i and j)
                    du_dij = self.a / (dij * (dij - self.a))

                    # calculate phi gradient
                    grad = - 2 * alpha[0] * np.array([xi, yi, zi * alpha[1]])

                    # update term 2
                    term_2 += np.dot(grad, (ri - rj)) * du_dij / dij

                    # calculate u second derivative
                    ddu_dij = (self.a * (self.a - 2 * dij)) / (dij ** 2 - dij * self.a) ** 2

                    # update term 4
                    term_4 += ddu_dij + (2 / dij) * du_dij

                    for k in range(len(particle_positions)):

                        if i == k:
                            continue
                        else:

                            # k vector
                            rk = np.array([particle_positions[k, 0],
                                           particle_positions[k, 1],
                                           particle_positions[k, 2]])

                            # calculate Euclidean distance between particle i and k
                            dik = np.linalg.norm(particle_positions[i, :] - particle_positions[k, :])

                            # calculate u first derivative (distance between i and k)
                            du_dik = self.a / (dik * (dik - self.a))

                            # update term 3
                            term_3 += (np.dot(ri - rj, ri - rk) / (dij * dik)) * du_dij * du_dik

        return term_1 + 2 * term_2 + term_3 + term_4

    def calculate_laplacian_numerically(self, particle_positions, alpha, step_size=0.001):

        squared_step_size = step_size ** 2
        laplacian = 0
        for i in range(len(particle_positions)):
            for j in range(len(particle_positions[0,:])):

                particle_positions[i, j] += step_size
                positive_step = self.evaluate(particle_positions, alpha)

                particle_positions[i, j] -= 2 * step_size
                negative_step = self.evaluate(particle_positions, alpha)

                particle_positions[i, j] += step_size
                middle_step = self.evaluate(particle_positions, alpha)

                laplacian += (positive_step - 2 * middle_step + negative_step) / (middle_step * squared_step_size)

        return laplacian

    def calculate_derivative(self, particle_positions, alpha):

        r2_sum = 0
        z2_sum = 0
        correlation = 1
        for i in range(len(particle_positions)):
            r2 = 0
            for j in range(len(particle_positions[0, :])):
                if j == 2:
                    r2 += alpha[1] * (particle_positions[i, j] ** 2)
                else:
                    r2 += particle_positions[i, j] ** 2
            r2_sum += r2
            z2_sum += particle_positions[i, 2] ** 2

            # interaction contribution
            for j in range(i + 1, len(particle_positions)):
                r_ij = np.linalg.norm(particle_positions[i, :] - particle_positions[j, :])
                if r_ij <= self.a:
                    correlation *= 0
                    return 0
                else:
                    correlation *= 1 - (self.a / r_ij)

        alpha_derivative = - r2_sum * np.exp(- alpha[0] * r2_sum) * correlation
        beta_derivative = - alpha[0] * z2_sum * np.exp(- alpha[0] * r2_sum) * correlation
        #beta_derivative = 0  # use if beta is fixed

        return np.array([alpha_derivative, beta_derivative])