import sys
import time
import random
import numpy as np
from array import *
from matplotlib import pyplot as plt
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	

random.seed(420)

N = 50
t = 900

evenGrid = np.zero([N, N], dtype=int)
oddGrid = np.zero([N, N]) dtype=int)

def update(timestep):
	for i in range(N):
		for j in range(N):

			#index calculation
			index = 0
			if(timestep % 2 == 0):
				index += evenGrid[(i - 1) % N, j] % 10
				index += evenGrid[(i + 1) % N, j] % 10
				index += evenGrid[i, (j + 1) % N] % 10
				index += evenGrid[i, (j - 1) % N] % 10
				index += evenGrid[(i - 1) % N, (j + 1) % N] % 10
				index += evenGrid[(i + 1) % N, (j - 1) % N] % 10
				index += evenGrid[(i + 1) % N, (j + 1) % N] % 10
				index += evenGrid[(i - 1) % N, (j - 1) % N] % 10
			else:
				index += oddGrid[(i - 1) % N, j] % 10
				index += oddGrid[(i + 1) % N, j] % 10
				index += oddGrid[i, (j + 1) % N] % 10
				index += oddGrid[i, (j - 1) % N] % 10
				index += oddGrid[(i - 1) % N, (j + 1) % N] % 10
				index += oddGrid[(i + 1) % N, (j - 1) % N] % 10
				index += oddGrid[(i + 1) % N, (j + 1) % N] % 10
				index += oddGrid[(i - 1) % N, (j - 1) % N] % 10
 			
			#state choices based on index and random number
			current_state = 0
			
