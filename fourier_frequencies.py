import torch
from torch.fft import fftfreq
from math import pi

# -----------------------------------------------
def FourierFrequencies(Nx:int, dx:float, Ny:int, dy:float, Nz:int, dz:float, 
                        device:torch.device=torch.device("cpu"), dtype=torch.float32):

    # in x-direction
    kx = (2.0 * pi * fftfreq(Nx, dx)).to(dtype).to(device)

    # in y-direction
    ky = (2.0 * pi * fftfreq(Ny, dy)).to(dtype).to(device)

    # in z-direction
    kz = (2.0 * pi * fftfreq(Nz, dz)).to(dtype).to(device)

    # Create grids of coordinates
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing = 'ij')

    # Concatenates a sequence of tensors
    freq = torch.stack((kx_grid, ky_grid, kz_grid))

    # Clean up
    del kx, ky, kz, kx_grid, ky_grid, kz_grid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return freq