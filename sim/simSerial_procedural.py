import sys
import time
import random
import numpy as np
from array import *
from pyinstrument import Profiler
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	

profiler = Profiler()
profiler.start()

random.seed(420)

N = 100
t = 900


#used later (larger scope for speed
state = 0
index = 0

evenGrid = np.zeros((N, N), dtype=int)
oddGrid = np.zeros((N, N), dtype=int)

def new_state(state, index):
	if state == 0:
		if random.randint(1, 501) < 2:
			return 2
		elif index < 7:
			return 0
		elif index < 17:
			return 1
		else:
			return 3

	elif state == 1:
		if random.randint(1, 501) < 2:
			return 3
		elif index < 1:
			return 0
		else:
			return 1
		
	elif state == 2:
		return random.choice([2,2,1,1,0])

	else:
		if index < 10:
			return 1
		else:
			return 3


def update(timestep):

	#for each lattice point
	for i in range(N):
		for j in range(N):

			#index calculation
			index = 0
			if timestep % 2 == 0:
				index += evenGrid[(i - 1) % N, j] 
				index += evenGrid[(i + 1) % N, j]
				index += evenGrid[i, (j + 1) % N] 
				index += evenGrid[i, (j - 1) % N] 
				index += evenGrid[(i - 1) % N, (j + 1) % N]
				index += evenGrid[(i + 1) % N, (j - 1) % N] 
				index += evenGrid[(i + 1) % N, (j + 1) % N] 
				index += evenGrid[(i - 1) % N, (j - 1) % N] 
			else:
				index += oddGrid[(i - 1) % N, j] 
				index += oddGrid[(i + 1) % N, j]
				index += oddGrid[i, (j + 1) % N]
				index += oddGrid[i, (j - 1) % N] 
				index += oddGrid[(i - 1) % N, (j + 1) % N] 
				index += oddGrid[(i + 1) % N, (j - 1) % N] 
				index += oddGrid[(i + 1) % N, (j + 1) % N] 
				index += oddGrid[(i - 1) % N, (j - 1) % N] 
 			
			#state choices based on index and random number
			if timestep % 2 == 0:
				state = evenGrid[i, j]
			else:
				state = oddGrid[i, j]

			state = new_state(state, index)

			#update future grid (depends on which timestep)
			if timestep % 2 == 0:
				oddGrid[i, j] = state
			else:
				evenGrid[i, j] = state


def printBoard(last_timestep):
	if last_timestep % 2 == 0:
		for i in range(N):
			for j in range(N):
				sys.stdout.write(str(str(evenGrid[i, j]) + " "))
			print("")
	else:
		for i in range(N):
			for j in range(N):
				sys.stdout.write(str(str(oddGrid[i, j]) + " "))
			print("")


for timestep in range(1, t+1):
	update(timestep)

profiler.stop()
printBoard(t)
print(profiler.output_text(unicode=True, color=True))
