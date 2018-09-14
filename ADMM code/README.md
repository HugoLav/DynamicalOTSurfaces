Implementation of the geodesic computation in the Wasserstein space on discrete surfaces as described in the paper Dynamical Optimal Transport on Discrete Surfaces

It reproduces the figures that can be found in the article. The parameters and the boundary conditions for each example are hard-coded in the files `example_****.py`. To run additional meshes, there is a `run_example.py` script that accepts arbitrary meshes (stored as .off files) and boundary conditions. Boundary conditions should be specified as executable Python scripts as in the .bdy files in `meshes/`

### Installation

To install, we recommend cloning the repository and installing the dependencies manually, or from the provided `requirements.txt` file.

```
git clone https://github.com/HugoLav/DynamicalOTSurfaces.git
cd ADMM\ code
pip install -r requirements
```

Dependencies: numpy, mayavi, matplotlib, scipy

### Usage

Example: 

To run a particular example, execute `example_****.py`:  
```
python example_airplane.py
```

To run on a new mesh (stored as `mesh.off`) with new boundary conditions (as `mesh.bdy`), execute `run_example.py`:
```
python run_example.py --mesh=mesh.off --boundary=mesh.bdy
```

The `run_example.py` script has flags that specify several of the variables from the paper.
