import sys
import time
import random
import numpy as np
from array import *
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

random.seed(420)

if(len(sys.argv) < 3):
	sys.exit("usage: python[3] simSerial_singleNeighborCalc.py [num lattice points each side of square] [num timesteps]") 
	time.sleep(1)

N = int(sys.argv[1])
t = int(sys.argv[2])


class Lattice:
	#static 2D array data structure for lattices
	dirt = []

	# states
 	# 0: dry
	# 1: saturated on top, no water on top
	# 2: not saturated, water on top
	# 3: saturated and water on top (ultra saturated)
	def __init__(self, x, y, state):
		self.x = x
		self.y = y
		self.state = state
		self.future = 0
		self.neighbors = []
		return

	def __str__(self):
		return str(self.state)

	def calcNeighbors(self, size):
		self.neighbors.append(Lattice.dirt[(self.x - 1) % size][(self.y - 1) % size])
		self.neighbors.append(Lattice.dirt[(self.x - 1) % size][(self.y + 1) % size])
		self.neighbors.append(Lattice.dirt[(self.x + 1) % size][(self.y - 1) % size])
		self.neighbors.append(Lattice.dirt[(self.x + 1) % size][(self.y + 1) % size])
		self.neighbors.append(Lattice.dirt[(self.x - 1) % size][self.y])
		self.neighbors.append(Lattice.dirt[(self.x + 1) % size][self.y])
		self.neighbors.append(Lattice.dirt[self.x][(self.y + 1) % size])
		self.neighbors.append(Lattice.dirt[self.x][(self.y - 1) % size])

	def computeIndex(self, size):
		index = 0
		for z in self.neighbors:
			index += z.state
		return index

	def update(self):
		self.state = self.future
		return

	def stageNextState(self, index, size):
		index = self.computeIndex(size)

		#completely dry
		#0.2% chance of getting rained on
		#index 0-6: stays dry
		#index 6-12: get saturated
		#index 13-24: gets ultra saturated
		if self.state == 0:
			if random.randint(1, 501) < 2:
				self.future = 2
			elif index < 7:
				self.future = 0
			elif index < 17:
				self.future = 1
			else:
				self.future = 3

		#saturated with no water on top
	   	#0.2% chance of ultra saturation (gets rained on)
		#index 0-2: future dries up
		#else: stays saturated 
		elif self.state == 1:
			if random.randint(1, 501) < 2 or index > 17:
				self.future = 3
			elif index < 1:
				self.future = 0
			else:
				self.future = 1

		#water on top with no saturation
		#40% chance of resting on top
		#40% chance of saturating earth
		#20% chance of drying up
		elif self.state == 2:
			r = random.choice([2, 2, 1, 1, 0])
			self.future = r
			
		#saturated with water on top	
		#index 0-9 goes to just saturated state
		#index 10-24 remains ultra-saturated
		else:
			if index < 10:
				self.future = 1
			else:
				self.future = 3


class Game:
	
	def __init__(self, size, timeSteps):
		self.size = size
		self.timeSteps = timeSteps
		self.lattices = Lattice.dirt
		return

	def setup(self):
		#print("Initializing...")
		for i in range(0, self.size):
			temp = []
			for j in range(0, self.size):	
				l = Lattice(i,j,0)
				temp.append(l)
			self.lattices.append(temp)
		#UNCOMMENT TO SEE LATTICE NEIGHBOR CALCULATION
		#count = 0
		for i in range(self.size):
			for j in range(self.size):
				self.lattices[i][j].calcNeighbors(self.size)
				#UNCOMMENT TO SEE LATTICE CURRENTLY WORKING
				#count += 1
				#print("Lattice point: " + str(count))
		return
	

	def stage(self):
		for i in range(self.size):
			for j in range(self.size):
				index = self.lattices[i][j].computeIndex(self.size)
				self.lattices[i][j].stageNextState(index, self.size)
		return
	
	def update(self):
		for i in range(self.size):
			for j in range(self.size):
				self.lattices[i][j].update()
		return

	def printPointAt(self, ex, why):
		sys.stdout.write(str(self.lattices[ex][why]) + " ") 
	
	def printBoard(self):
		for i in range(self.size):
			for j in range(self.size):
				self.printPointAt(i,j)
			print("")

	def play(self):
		self.setup()
		#print("Simulation beginning...")
		#time.sleep(1)		
		for i in range(1, self.timeSteps):
			self.stage()
			self.update()
			#UNCOMMENT IF YOU WANT TO SEE TIMESTEPS
			#print("Timestep: " + str(i))
		#print("Done! Printing board:")
		#time.sleep(1)
		profiler.stop()
		self.printBoard()
		#print("*" * self.size + str(i) + "*" * (self.size-1))
		return

a = Game(N, t)
a.play()
#print(profiler.output_text(unicode=True, color=True))
