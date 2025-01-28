import h5py
import numpy as np
from plots import PlotResults3D
import os
import sys

def ReadHDF5(results_directory):

    # Check if the .h5 file exists
    results_file = f"{results_directory}/results.h5"
    if not os.path.exists(results_file):
        print(f"Results .h5 file in {results_directory} does not exist.")
        sys.exit()

    with h5py.File(results_file, 'r') as hf_in:
        nsteps = hf_in['Simulation_Parameters/nsteps'][()]
        nt = hf_in['Simulation_Parameters/nt'][()]
        Nx = hf_in['Simulation_Parameters/Nx'][()]
        Ny = hf_in['Simulation_Parameters/Ny'][()]
        Nz = hf_in['Simulation_Parameters/Nz'][()]
        dx = hf_in['Simulation_Parameters/dx'][()]
        dy = hf_in['Simulation_Parameters/dy'][()]
        dz = hf_in['Simulation_Parameters/dz'][()]
        for step in (np.arange(0, nsteps+1, nt)):
            c = hf_in['Concentration/time_' + str(step)][()]
            PlotResults3D(results_directory, step, c, (Nx,Ny,Nz), (dx,dy,dz))
