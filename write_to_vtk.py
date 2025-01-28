"""
This Python script contains a function to write scalar field data to VTK files.

- Writes a VTK file with scalar field data.
- Parameters:
    - step (int): The timestep.
    - scalar_field (numpy.ndarray): The scalar field with shape (Nx, Ny, Nz).
    - dx (float): Grid spacing along the x-axis.
    - dy (float): Grid spacing along the y-axis.
    - dz (float): Grid spacing along the z-axis.
"""
import numpy as np

def WriteScalarToVTK(folder:str, step:int, scalar_field:np.array, dx:float, dy:float, dz:float):
    """
    Write a VTK file with scalar field data.
    """

    Nx, Ny, Nz = scalar_field.shape

    with open(f"{folder}/time_{str(step)}.vtk", 'w') as vtk_file:
        # Write VTK header
        vtk_file.write("# vtk DataFile Version 2.0\n")
        vtk_file.write("Scalar Field Data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET STRUCTURED_GRID\n")
        vtk_file.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        vtk_file.write(f"POINTS {Nx*Ny*Nz} float\n")

        # Write grid point coordinates
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    vtk_file.write(f"{i*dx} {j*dy} {k*dz}\n")

        # Write point data header
        vtk_file.write(f"POINT_DATA {Nx*Ny*Nz}\n")
        vtk_file.write("SCALARS scalar_field float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")

        # Write scalar field data
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    vtk_file.write(f"{scalar_field[i, j, k]}\n")