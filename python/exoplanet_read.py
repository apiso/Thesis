import numpy as np

f = open('../dat/ExoplanetData/transit.txt', 'r')
Mtransit, atransit = [], []
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
for line in f:
    Mtransit = np.append(Mtransit, float(line.split()[1]))
    atransit = np.append(atransit, float(line.split()[1]))