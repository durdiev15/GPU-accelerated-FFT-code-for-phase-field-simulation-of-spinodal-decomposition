"""
All data can be stored as .h5 file

Write_to_HDF5 -> in this function you can give some KEYWORD ARGUMENTS
"""

import h5py
import numpy as np

def WriteToHDF5(folder: str, step: int, data: np.array, **kwargs):
    # Open (or create) an HDF5 file in append mode
    hdf = h5py.File(str(folder) + '/results.h5', 'a')
    
    # Generate the time-specific dataset name
    time = '/time_' + str(int(step))

    # Create a dataset for the concentration data with a time-specific name
    hdf.create_dataset('Concentration' + str(time), data=data)

    # Loop over the keyword arguments to handle additional datasets
    for key, value in kwargs.items():
        if key == 'GradientEnergy':
            # Create a dataset for gradient energy if provided
            hdf.create_dataset('GradientEnergy' + str(time), data=value)
        elif key == 'DoubleWellPotential':
            # Create a dataset for double well potential if provided
            hdf.create_dataset('DoubleWellPotential' + str(time), data=value)
        # Save the simulation parameters
        elif key == 'SimulationParameters':
            sim_params = value # value must be dictionary 
            # Saving simulation parameters to HDF5
            sim_params_group = hdf.create_group('Simulation_Parameters')  # Group for simulation parameters
            sim_params_group.create_dataset('Nx', data=sim_params['Nx'])
            sim_params_group.create_dataset('Ny', data=sim_params['Ny'])
            sim_params_group.create_dataset('Nz', data=sim_params['Nz'])
            sim_params_group.create_dataset('dx', data=sim_params['dx'])
            sim_params_group.create_dataset('dy', data=sim_params['dy'])
            sim_params_group.create_dataset('dz', data=sim_params['dz'])
            sim_params_group.create_dataset('nsteps', data=sim_params['nsteps'])
            sim_params_group.create_dataset('nt', data=sim_params['nt'])
            sim_params_group.create_dataset('dt', data=sim_params['dt'])
            sim_params_group.create_dataset('mob', data=sim_params['mob'])
            sim_params_group.create_dataset('coefA', data=sim_params['coefA'])
            sim_params_group.create_dataset('k_grad', data=sim_params['k_grad'])

        """
        You can also add other if you have, for exampl:
        elif key == 'ElasticStrain':
            # Elastic strain tensor
            hdf.create_dataset('Elastic strain/strain_XX'+str(time), data = value[0,0])
            hdf.create_dataset('Elastic strain/strain_XY'+str(time), data = value[0,1])
            hdf.create_dataset('Elastic strain/strain_XZ'+str(time), data = value[0,2])
            hdf.create_dataset('Elastic strain/strain_YY'+str(time), data = value[1,1])
            hdf.create_dataset('Elastic strain/strain_YZ'+str(time), data = value[1,2])
            hdf.create_dataset('Elastic strain/strain_ZZ'+str(time), data = value[2,2])

        elif key == 'ElectricField':
            # Electric field tensor
            hdf.create_dataset('Electric field/Ex'+str(time), data = value[0])
            hdf.create_dataset('Electric field/Ey'+str(time), data = value[1])
            hdf.create_dataset('Electric field/Ez'+str(time), data = value[2])
        """

    # Close the HDF5 file to ensure all data is written and resources are freed
    hdf.close()
