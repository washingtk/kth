import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class SIS:

    # TODO : pypy/cython/numba Optimizing!

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

    def __init__(self, n, y, station_loc, t=500, mean_x=mean_x, cov_x=cov_x, z_candidate=z_candidate):
        self.n = int(n)
        self.t = int(t + 1)
        self.y = y
        self.station_loc = station_loc
        self.weight_history = np.zeros(shape=(self.n, self.t))
        self.tau = np.zeros(shape=(2, self.t))

        self.x = np.random.multivariate_normal(mean=mean_x, cov=cov_x, size=self.n).T
        self.move = np.random.randint(0, z_candidate.shape[1]-1, self.n)
        self.z = z_candidate[..., self.move]
        self.w = self.weight(i=0)
        self.weight_history[..., 0] = self.w
        self.tau[..., 0] = np.sum(self.x[[0, 3], ...] * self.w, axis=1) / sum(self.w)

    def start_explore(self):
        for i in range(1, self.t):
            self.next_x()
            self.next_z()
            self.w = self.weight(i=i)
            self.weight_history[..., i] = self.w
            if sum(self.w) != 0:
                self.tau[..., i] = np.sum(self.x[[0, 3], ...] * self.w, axis=1) / sum(self.w)
            else:
                pass

    def next_x(self, phi=phi, psi_z=psi_z, psi_w=psi_w, mean_w=mean_w, cov_w=cov_w):
        w = np.random.multivariate_normal(mean=mean_w, cov=cov_w, size=self.n).T
        self.x = phi@self.x + psi_z@self.z + psi_w@w

    def next_z(self, prob_z=prob_z, z_candidate=z_candidate):
        for i in range(0, self.n):
            self.move[i] = np.argmax(np.random.multinomial(1, prob_z[self.move[i], ...]))
        self.z = z_candidate[..., self.move]

    def weight(self, i, v=v, eta=eta, cov_v=cov_v):
        pdf = np.zeros(self.n)
        dist = self.distance_2d()
        if i == 0:
            for j in range(0, self.n):
                pdf[j] = multivariate_normal.pdf(self.y[:, i], mean=v - 10 * eta * np.log10(dist[..., j]), cov=cov_v)
        else:
            for j in range(0, self.n):
                pdf[j] = self.w[j] * multivariate_normal.pdf(self.y[:, i],
                                                 mean=v - 10 * eta * np.log10(dist[..., j]),
                                                 cov=cov_v)
        return pdf

    def distance_2d(self):
        x_2d = self.x[[0, 3], ...]
        dist = np.zeros(shape=(6, self.n))
        for i in range(0, self.n):
            dist[..., i] = np.sqrt(np.sum((self.station_loc.T - x_2d[..., i])**2, axis=1))
        return dist

    def important_weight(self):
        iw = np.zeros_like(self.weight_history)
        for i in range(0, self.t):
            if sum(self.weight_history[..., i]) != 0:
                iw[..., i] = self.weight_history[..., i] / sum(self.weight_history[..., i])
            else:
                break
        return iw

    def efficient_size(self):
        iw = self.important_weight()
        cv_square = np.zeros(self.t)
        ess = np.zeros(self.t)
        for i in range(0, self.t):
            cv_square[i] = np.sum((self.n * iw[..., i] - 1)**2, axis=0) / self.n
            if cv_square[i] != 1:
                ess[i] = self.n / (1 + cv_square[i])
            else:
                ess[i] = 0
        return ess

    def plot(self):
        plt.figure(1)
        plt.plot(self.station_loc[0, ...], self.station_loc[1, ...], 'o', label="station location")
        plt.plot(self.tau[0, ...], self.tau[1, :], '*', label="x tracking")
        plt.legend()
        plt.title("expected x-trajectory")

        ess = self.efficient_size()
        plt.figure(2)
        plt.plot(ess, label="ess")
        plt.grid()
        plt.legend()
        plt.xlabel("time step")
        plt.ylabel("sample N")
        plt.title("efficient sample size for each time step")
        plt.show()


class SISR(SIS):

    # FIXME : something might be wrong

    def start_explore(self):
        for i in range(1, self.t):
            self.resample_x()
            self.resample_z()
            self.next_x()
            self.next_z()
            self.w = self.weight(i=i)
            self.weight_history[..., i] = self.w
            if sum(self.w) != 0:
                self.tau[..., i] = np.sum(self.x[[0, 3], ...] * self.w, axis=1) / sum(self.w)
            else:
                pass


    def resample_x(self):
        for i in range(0, self.n):
            self.x[..., i] = self.x[..., np.argmax(np.random.multinomial(1, self.w))]

    def resample_z(self):
        for i in range(0, self.n):
            self.z[..., i] = self.z[..., np.argmax(np.random.multinomial(1, self.w))]

    def weight(self, i):
        pdf = np.zeros(self.n)
        dist = self.distance_2d()
        for j in range(0, self.n):
            pdf[j] = multivariate_normal.pdf(self.y[:, i],
                                             mean=self.v - 10 * self.eta * np.log10(dist[..., j]), cov=self.cov_v)
        return pdf