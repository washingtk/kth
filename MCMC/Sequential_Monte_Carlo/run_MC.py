import numpy as np
import Sequential_Monte_Carlo as smc
import generate_data as gd
import matplotlib.pyplot as plt

# gd.GD(station_loc=station_loc).generate_data()
x = np.loadtxt('generated_data/x_sample', delimiter=',')
y = np.loadtxt('generated_data/y_sample', delimiter=',')
station_loc = np.loadtxt('HA1-data/stations.txt', delimiter=',')

# track = smc.SIS(n=1e2, y=y, station_loc=station_loc)
# track.start_explore()
# track.plot()

track_with_resample = smc.SISR(n=1e3, y=y, station_loc=station_loc)
track_with_resample.start_explore()
track_with_resample.plot()


plt.figure()
plt.plot(x[0, ...], x[3,...], '*')
plt.show()