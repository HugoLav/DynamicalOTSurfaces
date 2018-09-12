"""
Produce the interpolation which corresponds to Figure 3 of the article
"""

import os

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl
from numpy import linalg as lin

from math import *

# Import the useful routines
import read_off
import surface_pre_computations
import geodesic_surface_congested
import cut_off

# To plot graph
import matplotlib.pyplot as plt

# To plot triangulated surfaces
from mayavi import mlab
from mayavi.api import Engine


engine = Engine()
engine.start()



# -----------------------------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------------------------

# Discretization of the starting [0,1] (for the centered grid)
nTime = 31

# Name of the file in which is stored the triangulated surface D
nameFileD = os.path.join("meshes", "square_regular_100.off")

# Parameter epsilon to regularize the Laplace problem
eps = 0.0*10**(-8)

# Number of iterations
Nit = 1000

# Detailed Study: True if we compute the objective functional at every time step (slow), False in the case we compute every 10 iterations.
detailStudy = False

# Value for the congestion parameter (alpha in the article)
cCongestion = 0.0

# -----------------------------------------------------------------------------------------------
# Read the .off file
# -----------------------------------------------------------------------------------------------

# Extract Vertices, Triangles, Edges
Vertices, Triangles, Edges = read_off.readOff(nameFileD)

# Compute areas of Triangles
areaTriangles, angleTriangles, baseFunction	 = surface_pre_computations.geometricQuantities(Vertices, Triangles, Edges)

# Compute the areas of the Vertices
originTriangles, areaVertices, vertexTriangles = surface_pre_computations.trianglesToVertices( Vertices,Triangles, areaTriangles)

# -----------------------------------------------------------------------------------------------
# Define the boundary conditions
# -----------------------------------------------------------------------------------------------

nVertices = Vertices.shape[0]

mub0 = np.zeros(nVertices)
mub1 = np.zeros(nVertices)

# Center of the "blob" for mub0
x0 = np.array([0.33,0.5,0.0])
# Center of the "blob" for mub1
x10 = np.array([0.8,0.2,0.0])
x11 = np.array([0.8,0.8,0.0])

for i in range(nVertices) :

	mub0[i] = areaVertices[i] * cut_off.f(lin.norm(Vertices[i,:] - x0) - 0.1 ,0.1)
	mub1[i] += areaVertices[i] * cut_off.f((lin.norm(Vertices[i,:] - x10) - 0.1)*2. ,0.1)
	mub1[i] += areaVertices[i] * cut_off.f((lin.norm(Vertices[i,:] - x11) - 0.1)*2. ,0.1)

# Normalization
mub0 /= np.sum(mub0)
mub1 /= np.sum(mub1)



# -----------------------------------------------------------------------------------------------
# Call the algorithm
# -----------------------------------------------------------------------------------------------

phi,mu,A,E,B,objectiveValue,primalResidual,dualResidual = geodesic_surface_congested.geodesic( nTime, nameFileD, mub0,mub1, cCongestion,eps, Nit, detailStudy )

# Plot the primal and dual residuals to check that everything is ok

plt.semilogy(primalResidual, label = "Primal residual")
plt.semilogy(dualResidual, label = "Dual residual")

plt.title("Primal and dual Residuals")

plt.legend()
plt.show()

# -----------------------------------------------------------------------------------------------
# Saving the results
# -----------------------------------------------------------------------------------------------

np.savetxt("plane_two_disks_mu_store.txt", mu.reshape((nTime*nVertices)) )

# -----------------------------------------------------------------------------------------------
# Printing the results with mayavi
# -----------------------------------------------------------------------------------------------

# Parameter for the mayavi scene of this example

def parametersMayavi(scene) :

	# Put the right angle
	scene.scene.z_minus_view()

	# Background color
	scene.scene.background = (1.0, 1.0, 1.0)

for i in [0,(nTime-1) // 4,  (nTime-1) // 2, (3*nTime-1) // 4, nTime-1] :

	# What we want to plot is the density
	muNormalized = np.divide(mu[i,:], areaVertices)

	# The normalization of the color map is done independently for each instant
	toPlot = (np.max(muNormalized) - muNormalized) / np.max(muNormalized)

	mlab.triangular_mesh(Vertices[:,0], Vertices[:,1], Vertices[:,2], Triangles, scalars=toPlot, colormap = "bone")

	parametersMayavi(engine.scenes[0])

	# Uncomment to save the figure
	# mlab.savefig("plane_two_disks_mu_" + str(i)+ ".png", size = (500,500))
	# mlab.close()

	mlab.show()
