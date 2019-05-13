import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

obs = np.loadtxt('mixture-observations.csv', delimiter=',')
plt.figure(1)
plt.hist(obs, label="distribution of obs")
plt.legend()
plt.show()

iteration = 50
theta = np.zeros(iteration)
theta[0] = 0.5
for i in range(1, iteration):
    theta[i] = np.sum((theta[i-1]*norm.pdf(obs, 1, 2) / (theta[i-1]*norm.pdf(obs, 1, 2) + (1-theta[i-1])*norm.pdf(obs, 0, 1)))) / len(obs)
plt.figure(2)
plt.plot(theta, label="theta updating")
plt.grid()
plt.legend()
plt.show()