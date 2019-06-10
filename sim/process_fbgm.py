import math
import os
import numpy as np
import mpi4py as MPI
from hdf5_4_CA import *
from mpi4py import MPI

conditions = open("conditions.fbgm", "r")
N = int(conditions.readline())
timesteps = int(conditions.readline())
size = int(conditions.readline())
height = int(conditions.readline())
dims = (1, N, N)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#change to something divisible by timesteps

for timestep in range(1, int(timesteps/size)+1):
	grid = np.fromfile("./data/end" + str((timestep*size)-int(rank)) + ".fbgm", dtype=int)
	state_data = np.empty((N*N), dtype=int)
	rank_data = np.empty((N*N), dtype=int)
	row = 0
	col = 0

	for i in range(N*N):
		state_data[i] = grid[i]
		rank_data[i] = int(row/height)
		col += 1
		if col == N:
			col = 0
			row += 1

	fname_h5 = "CA" + str((timestep*size) - int(rank)) + ".h5"
	fname_xmf = "CA" + str((timestep*size) - int(rank)) + ".xmf"


	writeH5(state_data, rank_data, fname_h5)
	writeXdmf(dims, N, fname_xmf, fname_h5)


