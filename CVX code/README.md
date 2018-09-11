Implementation of the convex cone problem from Equation (25) in the paper Dynamical Optimal Transport on Discrete Surfaces in CVX.

This is an alternative implementation to the ADMM implementation.

### Requirements

The code has been tested on MATLAB R2018a, R2017b, R2017a. May not work with versions earlier than R2014.

### Usage

The main solver is run from the `socpRun.m` script which takes as input `.mat` files that contain mesh and boundary information. Several such `.mat` files are saved under `meshes/`.

To run a particular example, call
```
[mu, runtime] = socpRun(dataFile, N, alpha)
```
The values in `mu` represent interpolants of the boundary probability measures. These values are not normalized; for probability distributions, multiply by area weights `mu .* M.areaWeights` where `M` is a mesh data structure (see `getMeshData.m`).