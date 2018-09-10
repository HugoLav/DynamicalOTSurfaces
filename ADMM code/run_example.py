"""Runs the interpolation algorithm on a given mesh and boundary conditions.

The boundary distributions are Python scripts that are executed locally by
the script. They can have arbitrary extensions, but must be valid Python code.

Example usage:
  python3 run_example.py --mesh=meshes/armadillo.off --boundary=armadillo.bdy
                         --alpha=0.0 --ntime=31

Command line arguments:
  mesh: Mesh file in .off format (required).
  boundary: Boundary distributions file. Should be valid Python (required).
  output: Text file to store the output intermediate distributions.
  alpha: Value of the congestion parameter.
  ntime: Number of time steps to consider. Typically an odd number.
  niter: Maximum number of ADMM iterations.
  verbose: Print out intermediate results. Setting this will slow down execution.
"""
import argparse
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
from geodesic_surface_congested import geodesic
import cut_off

parser = argparse.ArgumentParser(description='Process airplane model.')
parser.add_argument('-m', '--mesh', required=True,
                    help='relative path to mesh file')
parser.add_argument('-b', '--boundary', required=True,
                    help='relative path to boundary conditions file')
parser.add_argument('-o', '--output', default='output.log',
                    help='path to output file')
parser.add_argument('--alpha', default=0.0, type=float,
                    help='value of congestion parameter')
parser.add_argument('--ntime', default=31, type=int,
                    help='number of time steps for discretization')
parser.add_argument('--niter', default=1000, type=int,
                    help='maximum number of ADMM iterations')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='print intermediate costs')
args = parser.parse_args()
nameFileD = args.mesh
boundaryFileD = args.boundary
outputFileD = args.output
verbose = args.verbose

# Discretization of the starting [0,1] (for the centered grid)
nTime = args.ntime
# Parameter epsilon to regularize the Laplace problem
eps = 0.0 * 10 ** (-8)
# Number of iterations
Nit = args.niter
# Detailed Study: True if we compute the objective functional at every time step (slow),
# False in the case we compute every 10 iterations.
detailStudy = True
# Value for the congestion parameter (alpha in the article)
cCongestion = args.alpha

# Extract Vertices, Triangles, Edges
Vertices, Triangles, Edges = read_off.readOff(nameFileD)
# Compute areas of Triangles
areaTriangles, angleTriangles, baseFunction = surface_pre_computations.geometricQuantities(
    Vertices, Triangles, Edges
)
# Compute the areas of the Vertices
originTriangles, areaVertices, vertexTriangles = surface_pre_computations.trianglesToVertices(
    Vertices, Triangles, areaTriangles
)
nVertices = Vertices.shape[0]

# Define the boundary conditions
with open(boundaryFileD) as f:
    boundary = f.read()
    exec(boundary)

# Call the algorithm
phi, mu, A, E, B, objectiveValue, primalResidual, dualResidual = geodesic(
    nTime, nameFileD, mub0, mub1, cCongestion, eps, Nit, detailStudy, verbose
)

# Saving the results
np.savetxt(outputFileD, mu.reshape((nTime * nVertices)))
np.savetxt('output_phi.txt', phi.reshape(((nTime + 1) * nVertices)))
# np.savetxt(outputFileD, phi.reshape(((nTime + 1) * nVertices)))
