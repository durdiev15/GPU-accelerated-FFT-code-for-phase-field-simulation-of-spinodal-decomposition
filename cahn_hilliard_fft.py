"""
This is an example of how we can use PyTorch for GPU-accelerated phase-field simulations of spinodal decomposition (3D). 
The example is taken from the book Programming Phase-Field Modeling (Case Study 6, page 102). 
Here, we will use the FFT method to approximate the derivatives..
"""

import torch # ML library, but we will use its FFT functions
from torch.fft import fftn as fft, ifftn as ifft # Only FFT and IFFT needed
import os # Directory operations
import shutil # File manipulation

from fourier_frequencies import FourierFrequencies # Computes Fourier wave vectors
from write_to_hdf5 import WriteToHDF5 # Saves data to .h5 files
from write_to_vtk import WriteScalarToVTK # Saves data to .vtk files for ParaView
from energies import GradientEnergy, DoubleWellPotential # Calculates phase-field energies
from plots import PlotResults2D, PlotTotalEnergy # Plots 2D slices and total energy over time
from read_h5 import ReadHDF5 # Reads data from .h5 files for 3D plots

# Spinodal decomposition 
def CahnHilliard(folder:str, sim_params:dict, c:torch.tensor, device:torch.device):

    # Save initial data with simulation parameters  
    WriteToHDF5(folder=folder,
                step=0,
                data=c.to(torch.device("cpu")),
                SimulationParameters=sim_params) # for .h5
    WriteScalarToVTK(folder=folder,
                     step=0,
                     scalar_field=c.to(torch.device("cpu")),
                     dx=sim_params['dx'],
                     dy=sim_params['dy'],
                     dz=sim_params['dz']) # for .vtk

    # Fourier wave vectors
    freq = FourierFrequencies(Nx=sim_params['Nx'], dx=sim_params['dx'],
                              Ny=sim_params['Ny'], dy=sim_params['dy'],
                              Nz=sim_params['Nz'], dz=sim_params['dz'],
                              device=device)

    k2 = torch.sum(freq**2, dim=0)
    k4 = k2**2

    # Keep energies
    total_energy_data = []
    volume = sim_params['Nx'] * sim_params['Ny'] * sim_params['Nz']

    # Start the time loop 
    for step in range(sim_params['nsteps']):
        print(f"\n---------------------- Time step:\t {step}\t----------------------")

        # Derivative of hte double-well potential
        dfc_dc = sim_params['coefA'] * (c**2*(2*c - 2) + 2*c*(1 - c)**2)

        # Solve c
        ck = fft(c)

        # Energies
        grad_energy = GradientEnergy(ck, freq, sim_params['k_grad'])
        dw_potential = DoubleWellPotential(sim_params['coefA'], c)
        print(f"Gradient energy:     {torch.mean(grad_energy)*volume:.3e} [~]")
        print(f"DW potential energy: {torch.mean(dw_potential)*volume:.3e} [~]")

        ck = (ck - sim_params['dt'] * k2 * sim_params['mob'] * fft(dfc_dc)) / (1 + sim_params['dt'] * k4 * sim_params['mob'] * sim_params['k_grad'])
        c = ifft(ck).real

        # Save
        if (step + 1) % sim_params['nt'] == 0:
            WriteToHDF5(folder, step+1, c.to(torch.device("cpu")), GradientEnergy=grad_energy, DoubleWellPotential=dw_potential) # for .h5
            WriteScalarToVTK(folder, step+1, c.to(torch.device("cpu")), dx=sim_params['dx'], dy=sim_params['dy'], dz=sim_params['dz']) # for .vtk
            PlotResults2D(folder, step+1, c[:,:,sim_params['Nz']//2]) # 2D slice plots 

        total_energy_data.append(torch.mean(grad_energy + dw_potential)*volume)

        # Clean memory
        del dfc_dc, ck, grad_energy, dw_potential
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return total_energy_data

def main():
    
    # This is the directory where we will save data
    results_directory = os.getcwd() + "/results"

    # Check if the directory exists
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory) # remove it
        os.makedirs(results_directory) # create 

    # Check if the .h5 file exists
    if os.path.exists(f"{results_directory}/results.h5"):
        os.remove(f"{results_directory}/results.h5")  

    # Adding the simulation parameters into sim_params
    sim_params = {
        # Materials paraemters
        'mob': 1.0,
        'coefA': 1.0,
        'k_grad': 0.5,
        
        # Grid parameters
        'Nx': 64, 'Ny': 64, 'Nz': 64,
        'dx': 1, 'dy': 1, 'dz': 1,
        
        # Simulation time parameters
        'nsteps': 1000,
        'nt': 100,
        'dt': 5e-2    
    }

    # Choose device: "cpu" or "cuda"
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA
        # print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")  # Use CPU

    # Initial concentration
    torch.manual_seed(43)
    c=0.4 + 0.02 * (0.5 - torch.rand(sim_params['Nx'], sim_params['Ny'], sim_params['Nz']))


    # Run the simulation
    total_energy_data = CahnHilliard(folder=results_directory,
                        sim_params=sim_params,
                        c=c.to(device),
                        device=device)
    
    # Plot the total energy over time
    PlotTotalEnergy(results_directory, total_energy_data)

    # Read .h5 file
    ReadHDF5(results_directory)

if __name__ == "__main__":
    main()