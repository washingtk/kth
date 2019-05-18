import numpy as np
import matplotlib.pyplot as plt
import sampling as samp

coal_mine = np.loadtxt("coal-mine.csv", dtype=float)
plt.plot(coal_mine, 'g', legend="correlation between year and n_th disaster")
plt.scatter(range(1,192), coal_mine, alpha=0.3)
plt.title("trend transition of how frequently a disaster happened")
plt.xlabel("n_th disaster")
plt.ylabel("year")
plt.legend()
plt.show()

n = 1e3
burn_in = 1e2
d =5
nu = 1
rho = 0.1
hybrid_sample = samp.Sampling(d=d, nu=nu, rho=rho, n=n, burn_in=burn_in)
theta, lambdas, t = hybrid_sample.hybrid_sample()

plt.plot(theta)