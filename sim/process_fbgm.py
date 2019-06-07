import math
import os
import numpy as np
import mpi4py as MPI
from hdf5_4_CA import *
import sys


conditions = open("conditions.fbgm", "r")
N = int(conditions.readline())
timesteps = int(conditions.readline())
size = int(conditions.readline())
height = int(conditions.readline())

grid = np.fromfile("./end" + str(timesteps) + ".fbgm", dtype=int)
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

fname_h5 = "CA_1d_N=" + str(N) + "_t=" + str(timesteps) + "_s=" + str(size) + ".h5"
fname_xmf = "CA_1d_N=" + str(N) + "_t=" + str(timesteps) + "_s=" + str(size) + ".xmf"

dims = (1, N, N)

writeH5(state, rank, fname_h5)
writeXdmf(dims, N, fname_xmf, fname_h5)


