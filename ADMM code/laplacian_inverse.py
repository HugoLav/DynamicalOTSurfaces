# Clock
import time

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl
from numpy import linalg as lin

from math import *


def buildLaplacianMatrix(geomDic, eps):
    """Return a function which inverts the space-time Laplacian

    Args:
      geomDic: a dictionnary containing the relevant quantities concerning the space time domain
      eps: a parameter to regularize the pb, we compute the inverse of [Laplacian + esp * Identity]
    """

    # Unwrap what is needed in the dictionnary
    nTime = geomDic["nTime"]
    DeltaTime = geomDic["DeltaTime"]
    nVertices = geomDic["nVertices"]
    LaplacianDMatrix = geomDic["LaplacianDMatrix"]
    areaVertices = geomDic["areaVertices"]

    # Laplacian matrix in Time
    # Usual 1D Laplace equation

    LaplacianTimeMatrix = np.zeros((nTime + 1, nTime + 1))

    # Fill the interior
    for alpha in range(1, nTime):
        LaplacianTimeMatrix[alpha, alpha] = -2.0
        LaplacianTimeMatrix[alpha, alpha + 1] = 1.0
        LaplacianTimeMatrix[alpha, alpha - 1] = 1.0

        # Fill the upper left corner
    LaplacianTimeMatrix[0, 1] = 1.0
    LaplacianTimeMatrix[0, 0] = -1.0

    # Fill the lower right corner
    LaplacianTimeMatrix[-1, -2] = 1.0
    LaplacianTimeMatrix[-1, -1] = -1.0

    LaplacianTimeMatrix *= 1 / (DeltaTime ** 2)

    # Array of 1/sqrt(2) except for the first and last coefficient
    diagTimeMOH = 1 / sqrt(2) * np.ones(nTime + 1)
    diagTimeMOH[0] = 1.0
    diagTimeMOH[-1] = 1.0

    # Same as the previous matrix, but vectorized in nVertices
    diagTimeMOHVectorized = np.kron(diagTimeMOH, np.ones(nVertices)).reshape(
        (nTime + 1, nVertices)
    )

    # Diagonalizing in Time and factorizing in D ----------------------------------------

    startFact = time.time()

    print("Factorizing the Laplace matrix...")

    # Express the Laplacian in its new basis
    LaplacianTimeMatrixModified = np.dot(
        np.diag(diagTimeMOH), np.dot(LaplacianTimeMatrix, np.diag(diagTimeMOH))
    )

    # Compute the spectral decomposition of the Laplacian in Time
    eigenValTime, eigenVectTime = np.linalg.eigh(LaplacianTimeMatrixModified)

    # Prefactorizing the Laplace matrix
    # For each eigenvalue lambda_i, listFactor[i] contains a method to
    # solve (-lambda_i Id + Laplacian_D)x = b.
    listFactor = []

    for alpha in range(nTime + 1):
        factor = scspl.factorized(
            (
                3. * LaplacianDMatrix
                - eps * scsp.eye(nVertices)
                + eigenValTime[alpha] / 3. * scsp.diags([areaVertices], [0])
            ).tocsc()
        )
        listFactor.append(factor)

    def LaplacianAuxInvert(input):

        # Diagonalizing
        input_diag = np.array(np.dot(eigenVectTime.transpose(), input))

        # Solving for each line eigenvector

        solution = np.zeros((nTime + 1, nVertices))

        for alpha in range(nTime + 1):
            solution[alpha, :] = listFactor[alpha](input_diag[alpha, :])

            # Inverse diagonalization
        output = np.array(np.dot(eigenVectTime, solution))

        return output

    def LaplacianInvert(input):

        return np.multiply(
            diagTimeMOHVectorized,
            LaplacianAuxInvert(np.multiply(input, diagTimeMOHVectorized)),
        )

    endFact = time.time()

    print(
        "Factorizing the Laplace matrix: " + str(round(endFact - startFact, 2)) + "s."
    )

    return LaplacianInvert
