"""
Precompute the geometrics operators on a triangulated surface  
"""

# Clock
import time

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl
from numpy import linalg as lin

from math import *


def geometricQuantities(Vertices, Triangles, Edges) :

	"""
	Compute the geometric quantities associated to the mesh 
	Takes in argument Vertices, Triangles and Edges (for instance given by the readOff function)
	Returns: 
	- The array of the areas of the triangles 
	- The gradient of the hat functions 
	- The angles between the edges 
	"""
	
	nVertices = Vertices.shape[0]
	nTriangles = Triangles.shape[0]
	nEdges = Edges.shape[0]

	# To return 
	
	# Base function: gradient of the hat function 
	# First coordinate: triangle 
	# Second coordinate is the index of the vertex: the hat function is 1 on this vertex, 0 on the others 
	# Third coordinate corresponds to the component of gradient the vector in R^3 
	baseFunction = np.zeros((nTriangles, 3,3 ))

	# Areas 
	areaTriangles = np.zeros(nTriangles)
	
	# Angles
	# First coordinate is the triangle 
	# Second coordinate is the vertex on which the angle is computed
	angleTriangles = np.zeros((nTriangles, 3))
	
	# Loop over all Triangles 
	
	for i in range(nTriangles) : 
	
		# vij: vector between the vertices i and j. 	
		v01 = Vertices[ Triangles[i,1],:] - Vertices[ Triangles[i,0],:]
		v12 = Vertices[ Triangles[i,2],:] - Vertices[ Triangles[i,1],:]
		v20 = Vertices[ Triangles[i,0],:] - Vertices[ Triangles[i,2],:]
		
		# Area 
		areaTriangles[i] = lin.norm( np.cross(  v01, v12  )  ) / 2 
		
		# Angles via scalar product 
		angleTriangles[i,0] = acos(  np.dot( v01, - v20  ) / (lin.norm(v01)*lin.norm(v20) ))
		angleTriangles[i,1] = acos(  np.dot( v12, - v01  ) / (lin.norm(v12)*lin.norm(v01) ))
		angleTriangles[i,2] = acos(  np.dot( v20, - v12  ) / (lin.norm(v20)*lin.norm(v12) ))

		# Gradient of the hat functions (unnormalized) 
		baseFunction[i,0,:] = - v01 + ( v12 ) * (  np.dot( v01, v12 ) / np.dot( v12,v12  )    )
		baseFunction[i,1,:] = - v12 + ( v20 ) * (  np.dot( v12, v20 ) / np.dot( v20,v20  )    )
		baseFunction[i,2,:] = - v20 + ( v01 ) * (  np.dot( v20, v01 ) / np.dot( v01,v01  )    )
	
		# Normalization of the gradients 	
		baseFunction[i,0,:] /= lin.norm(baseFunction[i,0,:])**2
		baseFunction[i,1,:] /= lin.norm(baseFunction[i,1,:])**2
		baseFunction[i,2,:] /= lin.norm(baseFunction[i,2,:])**2
	
	return areaTriangles, angleTriangles, baseFunction	
	

def geometricMatrices(Vertices, Triangles, Edges, areaTriangles, angleTriangles, baseFunction ) :

		"""
		Compute the geometric operators associated to the mesh. More precisely it returns 
		- The Laplacian matrix (with the cotan) 
		- The gradient matrix [scalar function over vertices] -> [vector field over Triangles] 
		- The divergence matrix which is the adjoint of the previous one 
		"""
		
		nVertices = Vertices.shape[0]
		nTriangles = Triangles.shape[0]
		nEdges = Edges.shape[0]
		
		# Gradient matrix 
		# Its size (3*nTriangles) * nVertices. Takes a vector which represents a scalar field over vertices and return a vector field over triangles.
		
		gradientMatrix = scsp.coo_matrix(( baseFunction[:,0,0]  , ([3*i for i in range(nTriangles)], Triangles[:,0] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,1,0]  , ([3*i for i in range(nTriangles)], Triangles[:,1] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,2,0]  , ([3*i for i in range(nTriangles)], Triangles[:,2] )) , shape = (nTriangles*3,nVertices)).tocsr()
		
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,0,1]  , ([3*i+1 for i in range(nTriangles)], Triangles[:,0] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,1,1]  , ([3*i+1 for i in range(nTriangles)], Triangles[:,1] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,2,1]  , ([3*i+1 for i in range(nTriangles)], Triangles[:,2] )) , shape = (nTriangles*3,nVertices)).tocsr()

		gradientMatrix += scsp.coo_matrix(( baseFunction[:,0,2]  , ([3*i+2 for i in range(nTriangles)], Triangles[:,0] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,1,2]  , ([3*i+2 for i in range(nTriangles)], Triangles[:,1] )) , shape = (nTriangles*3,nVertices)).tocsr()
		gradientMatrix += scsp.coo_matrix(( baseFunction[:,2,2]  , ([3*i+2 for i in range(nTriangles)], Triangles[:,2] )) , shape = (nTriangles*3,nVertices)).tocsr()

		# Divergence matrix, MINUS the adjoint of the previous one 
		
		divergenceMatrix = - gradientMatrix.transpose()
		
		# Laplacian matrix, with the cotan formula 
		# We copy the cotan formula 
		
		# Wrt the vertex 0 in each triangle  
		
		LaplaceMatrix = scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,0]), np.sin(angleTriangles[:,0])), (Triangles[:,1] , Triangles[:,2])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,0]), np.sin(angleTriangles[:,0])), (Triangles[:,1] , Triangles[:,1])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,0]), np.sin(angleTriangles[:,0])), (Triangles[:,2] , Triangles[:,1])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,0]), np.sin(angleTriangles[:,0])), (Triangles[:,2] , Triangles[:,2])  ) , shape = (nVertices,nVertices)).tocsr()
		
		# Wrt the vertex 1 in each triangle 
		
		LaplaceMatrix += scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,1]), np.sin(angleTriangles[:,1])), (Triangles[:,2] , Triangles[:,0])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,1]), np.sin(angleTriangles[:,1])), (Triangles[:,2] , Triangles[:,2])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,1]), np.sin(angleTriangles[:,1])), (Triangles[:,0] , Triangles[:,2])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,1]), np.sin(angleTriangles[:,1])), (Triangles[:,0] , Triangles[:,0])  ) , shape = (nVertices,nVertices)).tocsr()
		
		# Wrt the vertex 2 in each triangle 
		
		LaplaceMatrix += scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,2]), np.sin(angleTriangles[:,2])), (Triangles[:,0] , Triangles[:,1])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,2]), np.sin(angleTriangles[:,2])), (Triangles[:,1] , Triangles[:,1])  ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( 0.5*  np.divide( np.cos(angleTriangles[:,2]), np.sin(angleTriangles[:,2])), (Triangles[:,1] , Triangles[:,0]) ) , shape = (nVertices,nVertices)).tocsr()
		
		LaplaceMatrix += scsp.coo_matrix(( - 0.5*  np.divide( np.cos(angleTriangles[:,2]), np.sin(angleTriangles[:,2])), (Triangles[:,0] , Triangles[:,0])  ) , shape = (nVertices,nVertices)).tocsr()
		
		return gradientMatrix, divergenceMatrix, LaplaceMatrix
		
def trianglesToVertices( Vertices, Triangles, areaTriangles ) : 

	"""
	Compute the matrices which enable to go from the triangles to the vertices: 
	- originTriangle: matrix which takes in argument a vector nTriangles*3 and which returns a vector nVertices: each value indexed by (a,v) (a for the triangle, v for the vertex in the triangle) is multiplied by the area of a and indexed by v.  
	- areaVertices: vector of size nVertices, the value is the sum of the area of the triangles which contain the vertex. 
	- vertexTriangles: matrix which takes a function defined over vertices and returns a vector of size nTriangles * 3: the same function, but indexed by (a,v) where a is a triangle and v the number of the vertex in the triangle. 
	"""
	
	nVertices = Vertices.shape[0]
	nTriangles = Triangles.shape[0]
	
	# listOrigin gives the vertex which corresponds to a (a,v) (a triangle, v number of the vertex in the triangle) once the array (a,v) has been reshaped in an array of size 3*nTriangles
	listOrigin = []
	
	arrayArea = np.zeros(3*nTriangles)
	
	for i in range(3*nTriangles) : 
		j = i % nTriangles 
		k = i // nTriangles 
		listOrigin.append( Triangles[j,k] )
		arrayArea[i] = areaTriangles[j]
	
	# Build the matrix originTriangles
	originTriangles = scsp.coo_matrix(( arrayArea , ( listOrigin , range(3*nTriangles)  )  ) , shape = (nVertices,3*nTriangles)).tocsr()

	# areaTriangles is straightforward to get 
	areaVertices = originTriangles.dot( np.ones(  3*nTriangles )  )
	
	# vertexTriangles is almost the adjoint of the first matrix, but without the weighting by the areas.
	vertexTriangles = scsp.coo_matrix(( np.ones(3*nTriangles) , ( range(3*nTriangles) , listOrigin  )  ) , shape = (3*nTriangles, nVertices)).tocsr()
			
	return originTriangles, areaVertices, vertexTriangles
	
	
	