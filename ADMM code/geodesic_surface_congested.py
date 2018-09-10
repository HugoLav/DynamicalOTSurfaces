""" 
Contains the function geodesic which consist in the implementation of the
Algorithm described in Section 4 of "Dynamical Optimal Transport on Discrete
Surfaces"
"""


# Package importation
# Clock
import time

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
from numpy import linalg as lin

from math import *

import read_off
from surface_pre_computations import geometricQuantities
from surface_pre_computations import geometricMatrices
from surface_pre_computations import trianglesToVertices
import laplacian_inverse

# Geodesic function


def geodesic(
        nTime,
        nameFileD,
        mub0,
        mub1,
        cCongestion,
        eps,
        Nit,
        detailStudy=False,
        verbose=False,
        tol=1e-6):
    """Implementation of the algorithm for computing the geodesic in the Wasserstein space. 

    Arguments. 
      nTime: Number of discretization points in time 
      nameFileD: Name of the .off file where the triangle mesh is stored 
      mub0: initial probability distribution 
      mub1: final probability distribution 
      cCongestion: constant for the intensity of the regularization 
        (alpha in the article)
      eps: regularization parameter for the linear system inversion 
      Nit: Number of iterations 
      detailStudy: True if the value of the objective functional (i.e. the
        Lagrangian) and the residuals are computed at every time step (slow),
        false if computed every 10 iterations (fast)

    Output.
      phi,mu,A,E,B: values of the relevant quantities in the interpolation,
        cf. the article for more details (Note: E is the momentum, denoted by m
        in the article) objectiveValue: evolution (in term of the number of
        iterations of the ADMM) of the objective value, ie the Lagrangian
      primalResidual,dualResidual: evolution (in term of the number of
        iterations of the ADMM) of primal and dual residuals, cf the article for
        details of their computation.
    """

    startImport = time.time()

    # Boolean value saying wether there is congestion or not
    isCongestion = cCongestion >= 10 ** (-10)

    if verbose:
        print(15 * "-" + " Parameters for the computation of the geodesic " + 15 * "-")
        print("Number of discretization points in time: " + str(nTime))
        print("Name of the mesh file: " + nameFileD)
        if isCongestion:
            print("Congestion parameter: " + str(cCongestion) + "\n")
        else:
            print("No regularization\n")

    # Time domain: staggered grid
    xTimeS = np.linspace(0, 1, nTime + 1)
    # Step Time
    DeltaTime = xTimeS[1] - xTimeS[0]
    # Time domain: centered grid
    xTimeC = np.linspace(DeltaTime / 2, 1 - DeltaTime / 2, nTime)
    # Domain D: call the routines
    Vertices, Triangles, Edges = read_off.readOff(nameFileD)
    areaTriangles, angleTriangles, baseFunction = geometricQuantities(
        Vertices, Triangles, Edges
    )
    gradientDMatrix, divergenceDMatrix, LaplacianDMatrix = geometricMatrices(
        Vertices, Triangles, Edges, areaTriangles, angleTriangles, baseFunction
    )
    originTriangles, areaVertices, vertexTriangles = trianglesToVertices(
        Vertices, Triangles, areaTriangles
    )

    np.savetxt("vertex_areas.txt", areaVertices)

    # Size of the domain D
    nVertices = Vertices.shape[0]
    nTriangles = Triangles.shape[0]
    nEdges = Edges.shape[0]

    # Vectorized quantities
    # Enable to call in parallel on [0,1] x D something which is only defined on D.

    # Vectorized arrays
    areaVectorized = np.kron(
        np.kron(np.ones(6 * nTime), areaTriangles), np.ones(3)
    ).reshape(nTime, 2, 3, nTriangles, 3)
    areaVerticesGlobal = np.kron(np.ones(nTime), areaVertices).reshape(
        (nTime, nVertices)
    )
    areaVerticesGlobalStaggerred = np.kron(np.ones(nTime + 1), areaVertices).reshape(
        (nTime + 1, nVertices)
    )

    # Vectorized matrices
    vertexTrianglesGlobal = scsp.kron(scsp.eye(nTime), vertexTriangles)
    originTrianglesGlobal = scsp.kron(scsp.eye(nTime), originTriangles)

    # Data structure with all the relevant informations. To be used as an argument of functions
    geomDic = {
        "nTime": nTime,
        "DeltaTime": DeltaTime,
        "Vertices": Vertices,
        "Triangles": Triangles,
        "Edges": Edges,
        "areaTriangles": areaTriangles,
        "gradientDMatrix": gradientDMatrix,
        "divergenceDMatrix": divergenceDMatrix,
        "LaplacianDMatrix": LaplacianDMatrix,
        "originTriangles": originTriangles,
        "areaVertices": areaVertices,
        "vertexTriangles": vertexTriangles,
        "nVertices": nVertices,
        "nTriangles": nTriangles,
        "nEdges": nEdges,
        "areaVerticesGlobal": areaVerticesGlobal,
        "areaVectorized": areaVectorized,
    }

    # Build the Laplacian matrix in space time and its inverse
    LaplacianInvert = laplacian_inverse.buildLaplacianMatrix(geomDic, eps)

    # Variable initialization
    # Primal variable phi.
    # Staggerred in Time, defined on the vertices of D
    phi = np.zeros((nTime + 1, nVertices))
    # Lagrange multiplier associated to mu.
    # Centered in Time and lives on the vertices of D
    mu = np.zeros((nTime, nVertices))
    # Momentum E, lagrange mutliplier.
    # Centered in Time, the second component is indicating on which side of
    # the temporal it comes from. Third component indicates the origine of
    # the triangle on which it is. Fourth component is the triangle. Last
    # component corresponds to the fact that we are looking at a vector of R^3.
    E = np.zeros((nTime, 2, 3, nTriangles, 3))

    # Primal Variable A, corresponds to d_t phi.
    # Same staggering pattern as mu
    A = np.zeros((nTime, nVertices))
    # Primal variable B, same pattern as E
    B = np.zeros((nTime, 2, 3, nTriangles, 3))
    # Lagrange multiplier associated to the congestion. If there is no
    # congestion, the value of this parameter will always stay at 0.
    lambdaC = np.zeros((nTime, nVertices))
    # Making sure the boundary values are normalized
    mub0 /= np.sum(mub0)
    mub1 /= np.sum(mub1)
    # Build the boundary term
    BT = np.zeros((nTime + 1, nVertices))
    BT[0, :] = -mub0
    BT[-1, :] = mub1

    # ADMM iterations
    # Value of the "augmentation parameter" for the augmented Lagragian problem (update dynamically)
    r = 1.

    # Initialize the array which will contain the values of the objective functional
    if detailStudy:
        objectiveValue = np.zeros(3 * Nit)
    else:
        objectiveValue = np.zeros((Nit // 10))

    # Initialize the arry which will contain the residuals
    primalResidual = np.zeros(Nit)
    dualResidual = np.zeros(Nit)

    # Main Loop
    for counterMain in range(Nit):
        if verbose:
            print(30 * "-" + " Iteration " +
                  str(counterMain + 1) + " " + 30 * "-")

        if detailStudy:
            objectiveValue[3 * counterMain] = objectiveFunctional(
                phi, mu, A, E, B, lambdaC, BT, geomDic, r, cCongestion, isCongestion
            )
        elif (counterMain % 10) == 0:
            objectiveValue[counterMain // 10] = objectiveFunctional(
                phi, mu, A, E, B, lambdaC, BT, geomDic, r, cCongestion, isCongestion
            )

        # Laplace problem
        startLaplace = time.time()
        # Build the RHS
        RHS = np.zeros((nTime + 1, nVertices))
        RHS -= BT * nTime
        RHS -= gradATime(mu, geomDic)
        RHS += np.multiply(
            r * gradATime(A + lambdaC,
                          geomDic), areaVerticesGlobalStaggerred / 3
        )
        # We take the adjoint wrt tp the scalar product weighted by areas, hence the multiplication by areaVectorized
        RHS -= divergenceD(E, geomDic)
        RHS += r * divergenceD(np.multiply(B, areaVectorized), geomDic)

        # Solve the system
        phi = 1. / r * LaplacianInvert(RHS)
        endLaplace = time.time()
        if verbose:
            print(
                "Solving the Laplace system: "
                + str(round(endLaplace - startLaplace, 2))
                + "s."
            )

        if detailStudy:
            objectiveValue[3 * counterMain + 1] = objectiveFunctional(
                phi, mu, A, E, B, lambdaC, BT, geomDic, r, cCongestion, isCongestion
            )

        # Projection over a convex set ---------------------------------------------------------
        # It projects on the set A + 1/2 |B|^2 <= 0. We reduce to a 1D projection, then use a Newton method with a fixed number of iteration.
        startProj = time.time()

        # Computing the derivatives of phi
        dTphi = gradTime(phi, geomDic)
        dDphi = gradientD(phi, geomDic)

        # Computing what there is to project
        toProjectA = dTphi + 3. / r * np.divide(mu, areaVerticesGlobal)
        toProjectB = dDphi + 1. / r * np.divide(E, areaVectorized)

        # bSquaredArray will contain
        # (sum_{a ~ v} |a| |B_{a,v}|**2) / (4*sum_{a ~ v} |a|)
        # for each vertex v
        bSquaredArray = np.zeros((nTime, nVertices))
        # square and sum to compute in account the eulcidean norm and the temporal average
        squareAux = np.sum(np.square(toProjectB), axis=(1, 4))
        # average wrt triangles
        bSquaredArray = originTrianglesGlobal.dot(
            squareAux.reshape(nTime * 3 * nTriangles)
        ).reshape((nTime, nVertices))
        # divide by the sum of the areas of the neighboring triangles
        bSquaredArray = np.divide(bSquaredArray, 4 * areaVerticesGlobal)
        # Value of the objective functional. For the points not in the convex, we want it to vanish.
        projObjective = toProjectA + bSquaredArray
        # projDiscriminating is 1 is the point needs to be projected, 0 if it is already in the convex
        projDiscriminating = (
            np.greater(projObjective, 10 ** (-16) *
                       np.ones((nTime, nVertices)))
        ).astype(float)

        # Newton method iteration
        # Value of the Lagrange multiplier. Initialized at 0, not updated if already in the convex set
        xProj = np.zeros((nTime, nVertices))
        counterProj = 0

        # Newton's loop
        while np.max(projObjective) > 10 ** (-10) and counterProj < 20:
            # Objective functional
            projObjective = (
                toProjectA
                + 6 * (1. + cCongestion * r) * xProj
                + np.divide(bSquaredArray, np.square(1 - xProj))
            )
            # Derivative of the ojective functional
            dProjObjective = 6 * (1. + cCongestion * r) - 2. * np.divide(
                bSquaredArray, np.power(xProj - 1, 3)
            )
            # Update of xProj
            xProj -= np.divide(
                np.multiply(projDiscriminating, projObjective), dProjObjective
            )
            counterProj += 1

        # Update of A
        A = toProjectA + 6 * (1. + cCongestion * r) * xProj
        # Update of lambda
        lambdaC = -6 * cCongestion * r * xProj
        # Transfer xProj, which is defined on vertices into something which is defined on triangles
        xProjTriangles = np.kron(
            vertexTrianglesGlobal.dot(xProj.reshape(nTime * nVertices)).reshape(
                (nTime, 3, nTriangles)
            ),
            np.ones(3),
        ).reshape((nTime, 3, nTriangles, 3))

        # Update of B
        B[:, 0, :, :, :] = np.divide(
            toProjectB[:, 0, :, :, :], 1. - xProjTriangles)
        B[:, 1, :, :, :] = np.divide(
            toProjectB[:, 1, :, :, :], 1. - xProjTriangles)

        # Print the info
        endProj = time.time()
        if verbose:
            print("Pointwise projection: " +
                  str(round(endProj - startProj, 2)) + "s.")
            print(
                str(counterProj)
                + " iterations needed; error committed: "
                + str(np.max(projObjective))
                + "."
            )

        if detailStudy:
            objectiveValue[3 * counterMain + 2] = objectiveFunctional(
                phi, mu, A, E, B, lambdaC, BT, geomDic, r, cCongestion, isCongestion
            )

        # Gradient descent in (E,muTilde), i.e. in the dual
        # No need to recompute the derivatives of phi
        # Update for mu
        mu -= r / 3 * np.multiply(areaVerticesGlobal, A + lambdaC - dTphi)
        # Update for E
        E -= r * np.multiply(areaVectorized, B - dDphi)

        # Compute the residuals
        # For the primal residual, just what was updated in the dual
        primalResidual[counterMain] = sqrt(
            (
                scalarProductFun(
                    A + lambdaC - dTphi,
                    np.multiply(A + lambdaC - dTphi, areaVerticesGlobal / 3.0),
                    geomDic,
                )
                + scalarProductTriangles(B - dDphi, B - dDphi, geomDic)
            )
            / np.sum(areaTriangles)
        )
        # For the residual, take the RHS of the Laplace system and conserve only
        # BT and the dual variables mu, E
        dualResidualAux = np.zeros((nTime + 1, nVertices))
        dualResidualAux += BT / DeltaTime
        dualResidualAux += gradATime(mu, geomDic)
        dualResidualAux += divergenceD(E, geomDic)

        dualResidual[counterMain] = r * sqrt(
            scalarProductFun(
                dualResidualAux,
                np.multiply(dualResidualAux,
                            areaVerticesGlobalStaggerred / 3.0),
                geomDic,
            )
            / np.sum(areaTriangles)
        )
        # Break early if residuals are small
        if primalResidual[counterMain] < tol and dualResidual[counterMain] < tol:
            break
        # Update the parameter r
        # cf. Boyd et al. for an explanantion of the rule
        if primalResidual[counterMain] >= 10 * dualResidual[counterMain]:
            r *= 2
        elif 10 * primalResidual[counterMain] <= dualResidual[counterMain]:
            r /= 2
        # Printing some results
        if verbose:
            if detailStudy:
                print(
                    "Maximizing in phi, should go up: "
                    + str(
                        objectiveValue[3 * counterMain + 1]
                        - objectiveValue[3 * counterMain]
                    )
                )
                print(
                    "Maximizing in A,B, should go up: "
                    + str(
                        objectiveValue[3 * counterMain + 2]
                        - objectiveValue[3 * counterMain + 1]
                    )
                )
                if counterMain >= 1:
                    print(
                        "Dual update: should go down: "
                        + str(
                            objectiveValue[3 * counterMain]
                            - objectiveValue[3 * counterMain - 1]
                        )
                    )
            print("Values of phi:")
            print(np.max(phi))
            print(np.min(phi))

            print("Values of A")
            print(np.max(A))
            print(np.min(A))

            print("Values of mu")
            print(np.max(mu))
            print(np.min(mu))

            print("Values of E")
            print(np.max(E))
            print(np.min(E))

            if isCongestion:
                print("Congestion")
                print(
                    scalarProductFun(lambdaC, mu, geomDic)
                    - 1
                    / (2. * cCongestion)
                    * scalarProductFun(
                        lambdaC, np.multiply(
                            lambdaC, areaVerticesGlobal / 3.), geomDic
                    )
                )
                print(
                    cCongestion
                    / 2.
                    * np.sum(np.divide(np.square(mu), 1 / 3. * areaVertices))
                    / nTime
                )

    # Print some informations at the end of the loop
    if verbose:
        print("Final value of the augmenting parameter: " + str(r))
        # Integral of mu wrt space (depends on time), should sum up to 1.
        intMu = np.sum(mu, axis=(-1))
        print("Minimal and maximal value of int mu")
        print(np.min(intMu))
        print(np.max(intMu))
        print("Maximal and minimal value of mu")
        print(np.min(mu))
        print(np.max(mu))

        dTphi = gradTime(phi, geomDic)
        dDphi = gradientD(phi, geomDic)
        print("Agreement between nabla_t,D and (A,B)")
        print(np.max(np.abs(dTphi - A)))
        print(np.max(np.abs(dDphi - B)))
    endProgramm = time.time()
    print(
        "Primal/dual residuals at end: {}/{}".format(
            primalResidual[counterMain],
            dualResidual[counterMain]
        )
    )
    print(
        "Congestion norm: {}".format(np.linalg.norm(lambdaC - cCongestion * mu))
    )
    print(
        "Objective value at end: {}".format(objectiveValue[counterMain // 10])
    )
    print(
        "Total number of iterations: {}".format(counterMain)
    )
    print(
        "Total time taken by the computation of the geodesic: "
        + str(round(endProgramm - startImport, 2))
        + "s."
    )
    return phi, mu, A, E, B, objectiveValue, primalResidual, dualResidual


# Scalar products
def scalarProductFun(a, b, geomDic):
    """Scalar product between functions
    """
    return np.sum(np.multiply(a, b)) / geomDic["nTime"]


def scalarProductTriangles(a, b, geomDic):
    """Scalar product weighted by the area of the Triangles 
    """
    return (
        np.sum(np.multiply(np.multiply(a, b), geomDic["areaVectorized"]))
        / geomDic["nTime"]
    )


# Differential, averaging and projection operators
def gradTime(input, geomDic):
    """Gradient wrt Time, ie temporal derivative 
    """
    output = (input[1:, :] - input[:-1, :]) / geomDic["DeltaTime"]
    return output


def gradATime(input, geomDic):
    """Minus adjoint of gradATime
    """
    inputSize = input.shape
    output = np.zeros((inputSize[0] + 1, inputSize[1]))

    output[1:-1, :] = (input[1:, :] - input[:-1, :]) / geomDic["DeltaTime"]
    output[0, :] = input[0, :] / geomDic["DeltaTime"]
    output[-1, :] = -input[-1, :] / geomDic["DeltaTime"]

    return output


def gradientD(input, geomDic):
    """Gradient wrt to the space variable 

    Takes something which has the staggering pattern of phi and returns the
    staggering patern of E,B
    """

    output = np.zeros((geomDic["nTime"], 2, 3, geomDic["nTriangles"], 3))

    # We need to transpose in order to multiply by gradientDMatrix independtly for each time
    outputAux = (
        geomDic["gradientDMatrix"]
        .dot(input.transpose())
        .transpose()
        .reshape((geomDic["nTime"] + 1, geomDic["nTriangles"], 3))
    )

    # Store for the 1st vertex of each triangle
    # First for the time before
    output[:, 0, 0, :, :] = outputAux[:-1, :, :]
    # Then for the time after
    output[:, 1, 0, :, :] = outputAux[1:, :, :]

    # Store for the 2nd vertex of each triangle
    output[:, 0, 1, :, :] = outputAux[:-1, :, :]
    output[:, 1, 1, :, :] = outputAux[1:, :, :]

    # Store for the 3rd vertex of each triangle
    output[:, 0, 2, :, :] = outputAux[:-1, :, :]
    output[:, 1, 2, :, :] = outputAux[1:, :, :]

    return output


def divergenceD(input, geomDic):
    """Minus adjoint of the previous operator 
    """
    output = np.zeros((geomDic["nTime"] + 1, geomDic["nVertices"]))

    inputAux = input.reshape((6 * geomDic["nTime"], 3 * geomDic["nTriangles"]))

    # Transpose and reshape to apply divergenceDMatrix independently for fixed first two indices
    outputAux = (
        geomDic["divergenceDMatrix"]
        .dot(inputAux.transpose())
        .transpose()
        .reshape((geomDic["nTime"], 2, 3, geomDic["nVertices"]))
    )

    # Then sum in both time and vertex
    output[:-1, :] += (
        outputAux[:, 0, 0, :] + outputAux[:, 0, 1, :] + outputAux[:, 0, 2, :]
    )
    output[1:, :] += (
        outputAux[:, 1, 0, :] + outputAux[:, 1, 1, :] + outputAux[:, 1, 2, :]
    )

    return output


# Definition of the function whose saddle point has to be found
def objectiveFunctional(
    phi, mu, A, E, B, lambdaC, BT, geomDic, r, cCongestion, isCongestion
):
    """Computation of the objective functional, ie the Lagrangian 
    """
    output = 0.0

    # Boundary term
    output += np.dot(phi[0, :], BT[0, :]) + np.dot(phi[-1, :], BT[-1, :])
    objective_nonreg = output
    # Computing the derivatives of phi
    dTphi = gradTime(phi, geomDic)
    dDphi = gradientD(phi, geomDic)

    # Lagrange multiplier mu
    output += scalarProductFun(A + lambdaC - dTphi, mu, geomDic)

    # Lagrange multiplier E.
    output += np.sum(np.multiply(B - dDphi, E)) / geomDic["nTime"]

    # Penalization in lambda, only in the case of congestion
    if isCongestion:
        norm_term = sum(geomDic["areaVertices"])
        congestion_cost = (
            1
            / (2. * cCongestion)
            * scalarProductFun(
                lambdaC,
                np.multiply(lambdaC, geomDic["areaVerticesGlobal"] / norm_term),
                geomDic,
            )
        )
        output -= congestion_cost
    # Penalty in A, phi
    output -= (
        r
        / 2.
        * scalarProductFun(
            A + lambdaC - dTphi,
            np.multiply(A + lambdaC - dTphi,
                        geomDic["areaVerticesGlobal"] / 3.),
            geomDic,
        )
    )

    # Penalty in B, phi.
    print('Objective/congestion: {}/{}'.format(objective_nonreg, congestion_cost))
    output -= r / 2. * scalarProductTriangles(B - dDphi, B - dDphi, geomDic)

    return output
