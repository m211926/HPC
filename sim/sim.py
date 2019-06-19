import sys
import random
import numpy as np
from array import *
from matplotlib import pyplot as plt
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	

N = 20
t = 20

class Lattice:
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
		Lattice.dirt.append(self)
		return

	def __str__(self):
		return str(self.state)

	def neighbors(self, size):
		a = [] 	
		lattices = Lattice.dirt
		a += [z for z in lattices if z.x == self.x and z.y == self.y + 1 % size]
		a += [z for z in lattices if z.x == self.x and z.y + 1 % size == self.y]
		a += [z for z in lattices if z.x + 1 % size == self.x and z.y == self.y] 
		a += [z for z in lattices if z.x == self.x + 1 % size and z.y == self.y]
		a += [z for z in lattices if z.x + 1 % size == self.x and z.y == self.y + 1 % size]
		a += [z for z in lattices if z.x + 1 % size == self.x and z.y + 1 % size == self.y]
		a += [z for z in lattices if z.x == self.x + 1 % size and z.y + 1 % size == self.y]
		a += [z for z in lattices if z.x == self.x + 1 % size and z.y == self.y + 1 % size]
		return a

	def computeIndex(self, size):
		a = self.neighbors(size)
		index = 0
		for z in a:
			index += z.state
		return index

	def update(self):
		self.state = self.future
		self.future = 0
		return

	def stageNextState(self, index, size):
		index = self.computeIndex(size)

		#completely dry
		#1% chance of getting rained on
		#index 0-8: stays dry
		#index 9-16: gets saturated
		#index 17-24: gets ultra saturated
		if self.state == 0:
			if random.randint(1, 101) < 2:
				self.future = 2
			elif index < 9:
				self.future = 0
			elif index < 17:
				self.future = 1
			else:
				self.future = 3

		#saturated with no water on top
	   	#1% chance of ultra saturation (gets rained on)
		#index 0-2: future dries up
		#else: stays saturated 
		elif self.state == 1:
			if random.randint(1, 101) < 2:
				self.future = 3
			elif index < 3:
				self.future = 0
			else:
				self.future = 1

		#water on top with no saturation
		#50% chance of getting evaporated
		#30% chance of saturating earth
		#5% chance of staying same (floats on top)
		#15% chance of ultra saturation 
		elif self.state == 2:
			r = random.randint(1,101)
			if r < 51:
				self.future = 0
			elif r < 81:
				self.future = 1
			elif r < 86: 
				self.future = 2
			else: 
				self.future = 3

		#saturated with water on top	
		#index 0-16: future only saturated
		#index 17-20: future same as present
		#index 21-24: future dried up (erosion? washed away? doesn't make too much practical sense)
		else:
			if index < 17:
				self.future = 1
			elif index < 21:
				self.future = 3
			else:
				self.future = 0

class Game:
	
	def __init__(self, size, timeSteps):
		self.size = size
		self.timeSteps = timeSteps
		self.lattices = Lattice.dirt
		return

	def setup(self):
		for i in range(self.size):
			for j in range(self.size):
				state = random.choice([0,1,2,3])
				Lattice(i,j,state)
		return
	

	def stage(self):
		#lattices = Lattice.dirt
		for lattice in self.lattices:
			index = lattice.computeIndex(self.size)
			lattice.stageNextState(index, self.size)
		return
	
	def update(self):
		#lattices = Lattice.dirt
		for lattice in self.lattices:
			lattice.update()
		return

	def play(self):
		self.setup()
		for i in range(1, self.timeSteps):
			self.stage()
			print("finished a stage")
			self.update()
			print("finished a step")
		return

	def printPointAt(self, ex, why):
		#lattices = Lattice.dirt
		for lattice in self.lattices:
			if lattice.x == ex and lattice.y == why:
				sys.stdout.write(str(lattice)) 
				return

	def xResults(self):
		#lattices = Lattice.dirt
		a = [[z.x] for z in self.lattices]
		return a

	def yResults(self):
                #lattices = Lattice.dirt
                a = [[z.y] for z in self.lattices]
                return a

	def stateResults(self):
                #lattices = Lattice.dirt
                a = [[z.state] for z in self.lattices]
                return a


a = Game(N, t)
a.play()

for y in range(N):
	for x in range(N):
		a.printPointAt(x, y)
	print("")

			
#x = a.xResults()
#y = a.yResults()
#state = a.stateResults()

#for ycoordinate in y:
#	for xcoordinate in x:
#		print(

#for j in range(t):
#	b = [] 
#	for i in range(size):
#		b.append([z[2][j] for z in a.results() if z[0] == i
#	plt.spy(b)
#	plt.savefig("time{" + j + "}.png")
