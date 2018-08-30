"""
Read a .off file 
Extract the list of vertices, triangles, and oriented edges 
"""

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl
from numpy import linalg as lin

from math import *


def readOff(nameFile) :

	"""
	Read a .off file 
	Take as an argument the name of the file 
	Return Vertices, Triangles, Edges, where: 
	Vertices is the list of the vertices coordinates; 
	Triangles is the list of indexes in the triangles; 
	Edges is the list of indexes of the oriented edges;
	"""

	fileOff = open(nameFile, 'r')

	# The first line is supposed to be "OFF"

	if fileOff.readline() != 'OFF\n' : 
		print("Error: not a valid .off file")
		return 0
		
	# The second line gives the number of vertices and triangles

	toRead = fileOff.readline().split()

	nVertices = int( toRead[0] )
	nTriangles = int( toRead[1] )
	nEdges = 3 * nTriangles

	# Initialize the arrays to be returned 

	Vertices = np.zeros((nVertices,3))
	Triangles = np.zeros((nTriangles,3), dtype = int)
	Edges = np.zeros((nEdges,2), dtype = int)

	# Read the whole file 

	# Count the number of vertices 
	counterV = 0
	# Count the number of triangles 
	counterT = 0 

	for line in fileOff.readlines() : 
		
		toRead = line.split()
		 
		if toRead[0] == "3" : 
			
			# Three numbers: corresponds to a triangle 
			
			# Fill Triangles and Edges 
			Triangles[counterT,0] = int(toRead[1])
			Triangles[counterT,1] = int(toRead[2])
			Triangles[counterT,2] = int(toRead[3])
			
			Edges[3*counterT,0] = int(toRead[1])
			Edges[3*counterT,1] = int(toRead[2])
			Edges[3*counterT+1,0] = int(toRead[2])
			Edges[3*counterT+1,1] = int(toRead[3])
			Edges[3*counterT+2,0] = int(toRead[3])
			Edges[3*counterT+2,1] = int(toRead[1])
			
			counterT += 1
			
		else : 
			
			# Correspond to a vertex 
		
			# Fill vertices 
			Vertices[counterV,0] = float(toRead[0])
			Vertices[counterV,1] = float(toRead[1])
			Vertices[counterV,2] = float(toRead[2])
			
			counterV += 1
		
	# Close the file 

	fileOff.close()

	return Vertices, Triangles, Edges


