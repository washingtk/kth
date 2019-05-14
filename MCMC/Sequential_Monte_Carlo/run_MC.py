import numpy as np
import Sequential_Monte_Carlo as smc
import generate_data as gd
import matplotlib.pyplot as plt

"""
first generate data by implementing gd.GD
tracking them by SIS or SISR
"""

# gd.GD(station_loc=station_loc).generate_data()
x = np.loadtxt('generated_data/x_sample', delimiter=',')
y = np.loadtxt('generated_data/y_sample', delimiter=',')
station_loc = np.loadtxt('HA1-data/stations.txt', delimiter=',')

# SIS
track = smc.SIS(n=1e3, y=y, station_loc=station_loc)
track.start_explore()
track.plot()

# SISR
# track_with_resample = smc.SISR(n=1e3, y=y, station_loc=station_loc)
# track_with_resample.init_r()
# track_with_resample.start_explore()
# track_with_resample.plot()

# original data
plt.figure()
plt.plot(station_loc[0, ...], station_loc[1, ...], 'o', label="station location")
plt.plot(x[0, ...], x[3, ...], '*', label="target track")
plt.legend()
plt.show()