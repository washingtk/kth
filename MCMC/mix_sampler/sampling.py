import numpy as np
import math


class Sampling:

    # TODO: RuntimeWarning: overflow encountered in double_scalars

    coal_mine = np.loadtxt("coal-mine.csv", dtype=np.float64)

    def __init__(self, d, nu, rho, n=1e4, burn_in=1e3):
        self.d = d
        self.nu = nu
        self.rho = rho
        self.N = int(n + burn_in)
        self.theta = np.ones(shape=self.N, dtype=np.float64)
        self.lambda_s = np.ones(shape=(self.d, self.N), dtype=np.float64)
        self.t = np.zeros(shape=((self.d + 1), self.N), dtype=np.float64)
        self.t[0, ...] = 1851
        self.t[self.d, ...] = 1963
        self.t[..., 0] = np.linspace(1851, 1963, self.d + 1, dtype=np.float64)

    def hybrid_sample(self):
        for i in range(self.N - 1):
            self.sample_theta(i=i)
            self.sample_lambda(i=i)
            self.sample_t(i=i)
        return self.theta, self.lambda_s, self.t

    def sample_theta(self, i):
        self.theta[i+1] = np.random.gamma(2 + 2 * self.d, self.nu + np.sum(self.lambda_s[..., i]))

    def sample_lambda(self, i, coal_mine=coal_mine):
        for j in range(self.d):
            n1 = coal_mine > self.t[j, i]
            n2 = self.t[j+1, i] > coal_mine
            num_disaster = np.sum(n1 & n2)
            self.lambda_s[j, i+1] = np.random.gamma(2 + num_disaster, self.theta[i+1] + self.t[j+1, i] - self.t[j, i])

    def sample_t(self, i):
        for j in range(1, self.d):
            candidate = self.random_walk(i=i, j=j)
            if np.random.uniform(0, 1, 1) < self.metropolis_hastings(i=i, j=j, candidate=candidate):
                self.t[j, i+1] = candidate
            else:
                self.t[j, i+1] = self.t[j, i]

    def random_walk(self, i, j):
        epsilon = self.t[j+1, i] - self.t[j-1, i+1]
        candidate = self.t[j, i] + self.rho * np.random.uniform(-epsilon, epsilon, 1)
        return float(candidate)

    def metropolis_hastings(self, i, j, candidate, coal_mine=coal_mine):
        num_d1_cand = np.sum((coal_mine > candidate) & (self.t[j + 1, i] > coal_mine))
        num_d2_cand = np.sum((coal_mine > self.t[j - 1, i + 1]) & (candidate > coal_mine))
        if num_d2_cand * num_d2_cand > 0:
            t1 = self.t[j + 1, i] - self.t[j, i]
            t2 = self.t[j, i] - self.t[j - 1, i + 1]
            t1_cand = self.t[j + 1, i] - candidate
            t2_cand = candidate - self.t[j - 1, i + 1]

            num_d1 = np.sum((coal_mine > self.t[j, i]) & (self.t[j + 1, i] > coal_mine))
            num_d2 = np.sum((coal_mine > self.t[j - 1, i + 1]) & (self.t[j, i] > coal_mine))

            t_term = (t1_cand * t2_cand) / (t1 * t2)
            exp_term = np.exp(-self.lambda_s[j, i+1] * (t1_cand - t1) - self.lambda_s[j-1, i+1] * (t2_cand - t2), dtype=np.float64)
            lambda_term1 = np.float64(self.lambda_s[j, i + 1] ** (num_d1_cand - num_d1))
            lambda_term2 = np.float64(self.lambda_s[j - 1, i + 1] ** (num_d2_cand - num_d2))
            return np.float64(t_term * exp_term * lambda_term1 * lambda_term2)
        else:
            return float(0)