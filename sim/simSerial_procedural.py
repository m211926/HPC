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
current_state = 0
next_state = 0
index = 0

evenGrid = np.zeros((N, N), dtype=int)
oddGrid = np.zeros((N, N), dtype=int)

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
				current_state = evenGrid[i, j]
			else:
				current_state = oddGrid[i, j]

			if current_state == 0:
				if random.randint(1, 501) < 2:
					next_state = 2
				elif index < 7:
					next_state = 0
				elif index < 17:
					next_state = 1
				else:
					next_state = 3

			elif current_state == 1:
				if random.randint(1, 501) < 2:
					next_state = 3
				elif index < 1:
					next_state = 0
				else:
					next_state = 1
		
			elif current_state == 2:
				r = random.choice([2,2,1,1,0])
				next_state = r

			else:
				if index < 10:
					next_state = 1
				else:
					next_state = 3

			#update future grid (depends on which timestep)
			if timestep % 2 == 0:
				oddGrid[i, j] = next_state
			else:
				evenGrid[i, j] = next_state


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
#printBoard(t)
print(profiler.output_text(unicode=True, color=True))
