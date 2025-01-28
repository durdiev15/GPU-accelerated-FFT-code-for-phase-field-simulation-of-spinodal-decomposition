# GPU-Accelerated FFT Code for Phase-Field Simulation of Spinodal Decomposition

This repository contains an implementation of GPU-accelerated phase-field simulations of spinodal decomposition in 3D using PyTorch. The code utilizes Fast Fourier Transform (FFT) methods to efficiently compute derivatives, making it suitable for large-scale simulations. The example is based on the Cahn-Hilliard equation for spinodal decomposition and comes from the book *Programming Phase-Field Modeling (Case Study 6, page 102)*.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Description](#code-description)
- [Simulation Parameters](#simulation-parameters)
- [Results](#results)
- [License](#license)

## Installation

To use this code, you will need to have Python installed along with PyTorch and several other dependencies. You can install the necessary dependencies using the following command:

```bash
pip install torch h5py vtk matplotlib numpy pyvista
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/durdiev15/GPU-accelerated-FFT-code-for-phase-field-simulation-of-spinodal-decomposition.git
    ```

2. Modify the simulation parameters in the `main()` function in `cahn_hilliard_fft.py` as desired. For instance, you can adjust the grid size, the number of time steps, and other material parameters.

3. Run the simulation by executing the following command:
    ```bash
    python cahn_hilliard_fft.py
    ```

4. The simulation will output results to a directory called `results` (created in the current working directory). It will also generate `.h5` files and `.vtk` files for further analysis and visualization.

5. After running the simulation, you can visualize the results using tools like ParaView to inspect the `.vtk`output files.

## Code Description

The code is structured as follows:

* **Cahn-Hilliard Function (CahnHilliard)**: This is the `main` function that runs the simulation. It computes the evolution of the concentration field `c` over time, using FFT to compute derivatives and update the concentration field in Fourier space. It also saves intermediate results in `.h5` and `.vtk` formats at specified time intervals.

* **Simulation Parameters**: The parameters for the simulation, including material properties (mobility, potential coefficients), grid size, and simulation time steps, are stored in the `sim_params` dictionary.

* **Supporting Functions**:

    * `FourierFrequencies`: Computes the Fourier wave vectors based on the grid size and grid spacing.
    * `WriteToHDF5`: Saves data to .h5 files for later analysis.
    * `WriteScalarToVTK`: Saves scalar fields as `.vtk` files for visualization in ParaView.
    * `GradientEnergy` and `DoubleWellPotential`: Compute the gradient energy and double-well potential energy terms in the Cahn-Hilliard model.
    * `PlotResults2D` and `PlotTotalEnergy`: Plot 2D slices of the concentration field and the total energy over time, respectively.
    * `ReadHDF5`: Reads and processes the saved `.h5` files for further visualization or analysis.

## Simulation Parameters

The simulation parameters are defined in the `sim_params` dictionary. Below are the main parameters you can configure:

* **Phase-Field Parameters**:

    * `mob`: Mobility coefficient (default: 1.0)
    * `coefA`: Coefficient for the double-well potential (default: 1.0)
    * `k_grad`: Gradient energy coefficient (default: 0.5)

* **Grid Parameters**:

    * `Nx`, `Ny`, `Nz`: Number of grid points along each axis (default: 64)
    * `dx`, `dy`, `dz`: Grid spacing along each axis (default: 1)

* **Simulation Time Parameters**:

    * `nsteps`: Total number of time steps for the simulation (default: 1000)
    * `nt`: Number of time steps between saving output files (default: 100)
    * `dt`: Time step size (default: 0.05)

You can modify these values to suit your specific simulation requirements.

## Results

The simulation results are saved in the `results` directory. At each time step, the following files are generated:

* **.h5 files**: Contain the concentration field `c`, gradient energy, and double-well potential energy at each saved time step.
* **.vtk files**: Contain scalar field data for visualization in ParaView.
* **Plots**: 2D slices of the concentration field at the middle of the grid and total energy plots are saved as images.

Note: Use .h5 if the grid points are very large (>128 in each direction).

## License

This code is released under the MIT License. See the  LICENSE  file for more details.