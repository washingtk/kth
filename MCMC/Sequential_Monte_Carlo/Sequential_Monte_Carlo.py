import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# from numba import jitclass, int8, float32, float64


class SIS:

    # TODO : pypy/cython/numba Optimizing!

    mean_x = np.zeros(shape=6)
    cov_x = np.diag([500, 5, 5, 200, 5, 5])

    prob_z = np.array([[16, 1, 1, 1, 1], [1, 16, 1, 1, 1], [1, 1, 16, 1, 1], [1, 1, 1, 16, 1], [1, 1, 1, 1, 16]],
                      dtype=float) / 20
    z_candidate = np.array([[0, 0], [3.5, 0], [0, 3.5], [0, -3.5], [-3.5, 0]]).T

    mean_w = np.zeros(shape=2)
    sigma_w = 0.5
    cov_w = sigma_w ** 2 * np.diag([1., 1.])

    v = 90
    eta = 3
    zeta = 1.5
    cov_v = zeta ** 2 * np.diag([1., 1., 1., 1., 1., 1.])

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
        self.weight_history = np.zeros(shape=(self.n, self.t), dtype=float)
        self.tau = np.zeros(shape=(2, self.t))

        self.x = np.random.multivariate_normal(mean=mean_x, cov=cov_x, size=self.n).T
        self.num_z_cand = z_candidate.shape[1] - 1
        self.move = np.random.randint(0, self.num_z_cand, self.n)
        self.z = z_candidate[..., self.move]
        self.w = self.weight_0(i=0)
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
        for i in range(self.n):
            self.move[i] = np.argmax(np.random.multinomial(1, prob_z[self.move[i], ...]))
        self.z = z_candidate[..., self.move]

    def weight_0(self, i, v=v, eta=eta, cov_v=cov_v):
        pdf = np.zeros(self.n)
        dist = self.distance_2d()
        for j in range(self.n):
            pdf[j] = multivariate_normal.pdf(self.y[:, i],
                                             mean=v - 10 * eta * np.log10(dist[..., j]),
                                             cov=cov_v)
        return pdf

    def weight(self, i, v=v, eta=eta, cov_v=cov_v):
        pdf = np.zeros(self.n)
        dist = self.distance_2d()
        for j in range(self.n):
            pdf[j] = self.w[j] * multivariate_normal.pdf(self.y[:, i],
                                             mean=v - 10 * eta * np.log10(dist[..., j]),
                                             cov=cov_v)
        return pdf

    def distance_2d(self):
        x_2d = self.x[[0, 3], ...]
        dist = np.zeros(shape=(6, self.n))
        for i in range(self.n):
            dist[..., i] = np.sqrt(np.sum((self.station_loc.T - x_2d[..., i])**2, axis=1))
        return dist

    def important_weight(self):
        iw = np.zeros_like(self.weight_history)
        for i in range(self.t):
            if sum(self.weight_history[..., i]) != 0:
                iw[..., i] = self.weight_history[..., i] / sum(self.weight_history[..., i])
            else:
                pass
        return iw

    def efficient_size(self):
        iw = self.important_weight()
        cv_square = np.zeros(self.t)
        ess = np.zeros(self.t)
        for i in range(self.t):
            cv_square[i] = np.sum((self.n * iw[..., i] - 1)**2, axis=0) / self.n
            if cv_square[i] != 1:
                ess[i] = self.n / (1 + cv_square[i])
            else:
                ess[i] = 0
        return ess

    def plot(self, target):
        plt.figure(1)
        plt.plot(self.station_loc[0, ...], self.station_loc[1, ...], 'o', label="station location")
        plt.plot(self.tau[0, ...], self.tau[1, :], '*', label="x tracking")
        if target.any() != None:
            plt.plot(target[0], target[3], '*', label="target track")
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

    def init_resample(self):
        self.resample()

    def start_explore(self):
        for i in range(1, self.t):
            self.next_x()
            self.next_z()
            self.w = self.weight_0(i=i)
            self.weight_history[..., i] = self.w
            if sum(self.w) != 0:
                self.tau[..., i] = np.sum(self.x[[0, 3], ...] * self.w, axis=1) / sum(self.w)
            else:
                pass
            self.resample()

    def resample(self):
        resample = np.zeros(self.n, dtype='int')
        for i in range(self.n):
            resample[i] = np.argmax(np.random.multinomial(1, self.w))
        self.x = self.x[..., resample]
        self.z = self.z[..., resample]


# spec = [
#     ('y', float32[:]), ('station_loc', float32[:]), ('weight_history', float64[:]), ('tau', float32[:]),
#     ('x', float32[:]), ('num_z_cand', int8[:]), ('move', int8[:]), ('z', float32[:]), ('w', float64[:]),
#     ('mean_x', float32[:]), ('cov_x', float32[:]), ('prob_z', float32[:]), ('z_candidate', float32[:]),
#     ('mean_w', float32[:]), ('sigma_w', float32[:]), ('cov_w', float32[:]), ('v', float32),
#     ('eta', float32), ('zeta', float32), ('cov_v', float32), ('delta', float32), ('alpha', float32),
#     ('phi', float32[:]), ('psi_z', float32[:]), ('psi_w', float32[:])
# ]