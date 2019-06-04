import sys
import time
import random
import numpy as np
from array import *
from matplotlib import pyplot as plt
from math import ceil, floor, sqrt
#import plotly.plotly as py
#import plotly.tools as tls	

N = 50
t = 2000

class Lattice:
	#static array data structure for lattices
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
		self.neighbors = []
		return

	def __str__(self):
		return str(self.state)

	def calcNeighbors(self, size):
		a = []
		for z in Lattice.dirt:
			if z.x == self.x and z.y == self.y + 1 % size:
				a += [z]
			elif z.x == self.x and z.y + 1 % size == self.y:
				a += [z]
			elif z.x + 1 % size == self.x and z.y == self.y:
				a += [z]
			elif z.x == self.x + 1 % size and z.y == self.y:
				a += [z]
			elif z.x + 1 % size == self.x and z.y == self.y + 1 % size:
				a += [z]
			elif z.x + 1 % size == self.x and z.y + 1 % size == self.y:
				a += [z] 
			elif z.x == self.x + 1 % size and z.y + 1 % size == self.y:
				a += [z]
			elif z.x == self.x + 1 % size and z.y == self.y + 1 % size:
				a += [z]

		self.neighbors = a

	def computeIndex(self, size):
		index = 0
		for z in self.neighbors:
			index += z.state
		return index

	def update(self):
		self.state = self.future
		#self.future = 0
		return

	def stageNextState(self, index, size):
		index = self.computeIndex(size)

		#completely dry
		#1/3% chance of getting rained on
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
	   	#1/3% chance of ultra saturation (gets rained on)
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
		print("Initiallizing...")
		time.sleep(1)
		for i in range(self.size):
			for j in range(self.size):
				state = 0
				Lattice(i,j,state)
		#UNCOMMENT TO SEE LATTICE NEIGHBOR CALCULATION
		count = 0
		for l in self.lattices:
			l.calcNeighbors(self.size)
			#UNCOMMENT TO SEE LATTICE CURRENTLY WORKING
			count += 1
			print("Lattice point: " + str(count))
		return
	

	def stage(self):
		for lattice in self.lattices:
			index = lattice.computeIndex(self.size)
			lattice.stageNextState(index, self.size)
		return
	
	def update(self):
		for lattice in self.lattices:
			lattice.update()
		return

	def printPointAt(self, ex, why):
		for lattice in self.lattices:
			if lattice.x == ex and lattice.y == why:
				sys.stdout.write(str(lattice) + " ") 
				return
	
	def printBoard(self):
		for y in range(self.size):
			for x in range(self.size):
				self.printPointAt(x,y)
			print("")

	def play(self):
                self.setup()
                print("Simulation beginning...")
                time.sleep(1)		
                for i in range(1, self.timeSteps):
                        self.stage()
                        self.update()
                        #UNCOMMENT IF YOU WANT TO SEE TIMESTEPS
                        print("Timestep: " + str(i))
                print("Done! Printing board:")
                time.sleep(1)
                self.printBoard()
                #print("*" * self.size + str(i) + "*" * (self.size-1))
                return

a = Game(N, t)
a.play()

