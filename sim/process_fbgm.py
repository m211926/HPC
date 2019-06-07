import math
import os
import numpy as np
import mpi4py as MPI
from hdf5_4_CA import *


conditions = open("conditions.fbgm", "r")
N = int(conditions.readline())
timesteps = int(conditions.readline())
size = int(conditions.readline())
height = int(conditions.readline())
dims = (1, N, N)

for timestep in range(1, timesteps+1):
	grid = np.fromfile("./data/end" + str(timestep) + ".fbgm", dtype=int)
	state = np.empty((N*N), dtype=int)
	rank = np.empty((N*N), dtype=int)

	row = 0
	col = 0

	for i in range(N*N):
		state[i] = grid[i]
		rank[i] = int(row/height)
		col += 1
		if col == N:
			col = 0
			row += 1

	fname_h5 = "CA" + str(timestep) + ".h5"
	fname_xmf = "CA" + str(timestep) + ".xmf"


	writeH5(state, rank, fname_h5)
	writeXdmf(dims, N, fname_xmf, fname_h5)


