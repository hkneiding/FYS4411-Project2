from sys import argv
import numpy as np
import matplotlib.pyplot as plt


filename = argv[1]
positions = np.loadtxt(filename, delimiter=",")
distances = []

for i in range(positions.shape[0]):
    distances.append(np.sqrt(positions[i,0] ** 2 + positions[i,1] ** 2 + positions[i,2] ** 2))

mean_distance = sum(distances) / len(distances)

figure = plt.hist(distances, bins=200, color='b', normed=1)
plt.axvline(mean_distance, color='r', linestyle='dashed', label='Average distance')
plt.legend(loc=1)
plt.xlabel(r'$r/a_{ho}$')
plt.ylabel('Normalized number of particles')
plt.show()
