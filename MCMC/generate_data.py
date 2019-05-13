import numpy as np
from numpy.random import multivariate_normal


class GD:

    mean_x = np.zeros(shape=6)
    cov_x = np.diag([500, 5, 5, 200, 5, 5])

    prob_z = np.array([[16, 1, 1, 1, 1], [1, 16, 1, 1, 1], [1, 1, 16, 1, 1], [1, 1, 1, 16, 1], [1, 1, 1, 1, 16]]) / 20
    z_candidate = np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]]).T

    mean_w = np.zeros(shape=2)
    sigma_w = 0.5
    cov_w = sigma_w ** 2 * np.diag([1, 1])

    v = 90
    eta = 3
    zeta = 1.5
    cov_v = zeta ** 2 * np.diag([1, 1, 1, 1, 1, 1])

    delta = 0.5
    alpha = 0.6
    phi = np.array([[1, 0, 0, 0, 0, 0], [delta, 1, 0, 0, 0, 0], [delta ** 2, delta, alpha, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0], [0, 0, 0, delta, 1, 0], [0, 0, 0, delta ** 2, delta, alpha]]).T
    psi_z = np.array([[delta ** 2, delta, 0, 0, 0, 0], [0, 0, 0, delta ** 2, delta, 0]]).T
    psi_w = np.array([[delta ** 2, delta, 1, 0, 0, 0], [0, 0, 0, delta ** 2, delta, 1]]).T

    def __init__(self, station_loc, t=500, mean_x=mean_x, cov_x=cov_x, z_candidate=z_candidate):
        self.t = int(t + 1)
        self.station_loc = station_loc
        self.y_history = np.zeros(shape=(self.station_loc.shape[1], self.t))
        self.x_history = np.zeros(shape=(6, self.t))

        self.x_history[..., 0] = np.random.multivariate_normal(mean=mean_x, cov=cov_x)
        self.move = np.random.randint(0, z_candidate.shape[1]-1)
        self.z = z_candidate[..., self.move]
        self.calculate_y(i=0)

    def generate_data(self):
        for i in range(1, self.t):
            self.next_x(i=i)
            self.next_z()
            self.calculate_y(i=i)
        self.save_to_txt()

    def next_x(self, i, phi=phi, psi_z=psi_z, psi_w=psi_w, mean_w=mean_w, cov_w=cov_w):
        w = np.random.multivariate_normal(mean=mean_w, cov=cov_w)
        self.x_history[..., i] = phi@self.x_history[..., (i-1)] + psi_z@self.z + psi_w@w

    def next_z(self, prob_z=prob_z, z_candidate=z_candidate):
        self.move = np.argmax(np.random.multinomial(1, prob_z[self.move, ...]))
        self.z = z_candidate[..., self.move]

    def calculate_y(self, i, v=v, eta=eta, cov_v=cov_v):
        dist = self.distance_2d(i=i)
        self.y_history[..., i] = multivariate_normal(mean=v - 10 * eta * np.log10(dist), cov=cov_v)

    def distance_2d(self, i):
        x_2d = self.x_history[[0, 3], i]
        return np.sqrt(np.sum((self.station_loc.T - x_2d)**2, axis=1))

    def save_to_txt(self):
        np.savetxt('generated_data/x_sample', self.x_history, delimiter=',')
        np.savetxt('generated_data/y_sample', self.y_history, delimiter=',')