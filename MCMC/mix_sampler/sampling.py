import numpy as np


class Sampling:

    # TODO: RuntimeWarning: overflow encountered in double_scalars

    coal_mine = np.loadtxt("coal-mine.csv", dtype=float)

    def __init__(self, d, nu, rho, n=1e4, burn_in=1e3):
        self.d = d
        self.nu = nu
        self.rho = rho
        self.N = int(n + burn_in)
        self.theta = np.ones(shape=self.N, dtype=float)
        self.lambda_s = np.ones(shape=(self.d, self.N), dtype=np.float64)
        self.t = np.zeros(shape=((self.d + 1), self.N), dtype=float)
        self.t[0, ...] = 1851
        self.t[self.d, ...] = 1963
        self.t[..., 0] = np.linspace(1851, 1963, self.d + 1, dtype=float)

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
            if np.random.uniform(0, 1, 1) < (self.MH_for_cand(i=i, j=j, candidate=candidate) / self.MH_for_t(i=i, j=j)):
                self.t[j, i+1] = candidate
            else:
                self.t[j, i+1] = self.t[j, i]

    def random_walk(self, i, j):
        epsilon = self.t[j+1, i] - self.t[j-1, i+1]
        candidate = self.t[j, i] + self.rho * np.random.uniform(-epsilon, epsilon, 1)
        return float(candidate)

    def MH_for_t(self, i, j, coal_mine=coal_mine):
        t_1 = self.t[j+1, i] - self.t[j, i]
        t_2 = self.t[j, i] - self.t[j-1, i+1]
        num_d1 = np.sum((coal_mine > self.t[j, i]) & (self.t[j+1, i] > coal_mine))
        num_d2 = np.sum((coal_mine > self.t[j-1, i+1]) & (self.t[j, i] > coal_mine))
        exp_term = np.float32(np.exp(-self.lambda_s[j, i + 1] * t_1 - self.lambda_s[j - 1, i + 1] * t_2))
        lambda_term1 = np.float32(self.lambda_s[j, i+1] ** num_d1)
        lambda_term2 = np.float32(self.lambda_s[j - 1, i + 1] ** num_d2)
        return np.float32(t_1 * t_2 * exp_term * lambda_term1 * lambda_term2)

    def MH_for_cand(self, i, j, candidate, coal_mine=coal_mine):
        t_1 = self.t[j+1, i] - candidate
        t_2 = candidate - self.t[j-1, i+1]
        if t_1 * t_2 > 0:
            num_d1 = np.sum((coal_mine > candidate) & (self.t[j+1, i] > coal_mine))
            num_d2 = np.sum((coal_mine > self.t[j-1, i+1]) & (candidate > coal_mine))
            return t_1 * t_2 * np.exp(-self.lambda_s[j, i+1]*t_1-self.lambda_s[j-1, i+1]*t_2) * (self.lambda_s[j, i+1]**num_d1) * (self.lambda_s[j-1, i+1]**num_d2)
        else:
            return 0