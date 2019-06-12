import math
import os
import numpy as np
from hdf5_4_CA import *

conditions = open("conditions.fbgm", "r")
N = int(conditions.readline())
timesteps = int(conditions.readline())
size = int(conditions.readline())
height = int(conditions.readline())
dims = (1, N, N)

#change to something divisible by timesteps

for timestep in range(1, (timesteps / 100) + 1):
	grid = np.fromfile("./data/end" + str(timestep * 100) + ".fbgm", dtype=int)
	state_data = np.empty((N*N), dtype=int)
	row = 0
	col = 0

	for i in range(N*N):
		state_data[i] = grid[i]
		col += 1
		if col == N:
			col = 0
			row += 1

	fname_h5 = "CA" + str(timestep * 100) + ".h5"
	fname_xmf = "CA" + str(timestep * 100) + ".xmf"


	writeH5(state_data, fname_h5)
	writeXdmf(dims, N, fname_xmf, fname_h5)


