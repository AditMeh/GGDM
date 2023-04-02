import numpy as np
from matplotlib import pyplot as plt
import sys

data = np.load(sys.argv[1])
data = data > 0.3
colors = np.empty(data.shape, dtype=object)
colors[:, :, :] = 'blue'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(data, facecolors=colors, edgecolor='k')
plt.savefig("new.png")
