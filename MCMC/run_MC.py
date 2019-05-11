import Sequential_Monte_Carlo as smc

# track = smc.SIS(n=1e2)
# track.start_explore()
# track.plot()

track_with_resample = smc.SISR(n=1e2)
track_with_resample.start_explore()
track_with_resample.plot()