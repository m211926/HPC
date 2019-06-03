import sys
import random
import numpy as np
from array import *
from matplotlib import pyplot as plt
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	

pWidth = 500

N = 20
t = 2000

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
		#index 0-6: stays dry
		#index 6-12: get saturated
		#index 13-24: gets ultra saturated
		if self.state == 0:
			if random.randint(1, 301) < 2:
				self.future = 2
			elif index < 7:
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
			if random.randint(1, 301) < 2 or index > 17:
				self.future = 3
			elif index < 1:
				self.future = 0
			else:
				self.future = 1

		#water on top with no saturation
		#40% chance of resting on top
		#40% chance of saturating earth
		#10% chance of drying up
		elif self.state == 2:
			r = random.choice([2, 2, 1, 1, 0])
			self.future = r
			

		#saturated with water on top	
		#index 0-16: future only saturated
		#index 17-20: future same as present
		#index 21-24: future dried up (erosion? washed away? doesn't make too much practical sense)
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
		for i in range(self.size):
			for j in range(self.size):
				state = 0 #random.choice([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3])
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

	def printPointAt(self, ex, why):
		for lattice in self.lattices:
			if lattice.x == ex and lattice.y == why:
				sys.stdout.write(str(lattice)) 
				return
	
	def printBoard(self):
		for y in range(self.size):
			for x in range(self.size):
				self.printPointAt(x,y)
			print("")

	def play(self):
                self.setup()
                for i in range(1, self.timeSteps):
                        self.stage()
                        self.update()
                        self.printBoard()
                        print("*" * self.size)
                return

	#def xResults(self):
		#lattices = Lattice.dirt
		#a = [[z.x] for z in self.lattices]
		#return a

	#def yResults(self):
                #lattices = Lattice.dirt
                #a = [[z.y] for z in self.lattices]
                #return a

	#def stateResults(self):
                #lattices = Lattice.dirt
                #a = [[z.state] for z in self.lattices]
                #return a

a = Game(N, t)
a.play()

			
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
