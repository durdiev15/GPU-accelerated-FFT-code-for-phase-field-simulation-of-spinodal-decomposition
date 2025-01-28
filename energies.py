import torch
from torch.fft import fftn as fft, ifftn as ifft

def GradientEnergy(c_k, freq, k_grad):
    
    # Compute gradients in Fourier space
    dc_k = 1j * freq * c_k # dc/dx_i in Fourier space
    # , dim=(1,2,3)

    # Transform gradients back to real space
    dc = ifft(dc_k, dim=(1,2,3)).real

    # Compute the gradient energy
    return 0.5 * k_grad * torch.sum(dc**2, dim=0)

def DoubleWellPotential(coefA, c):
    # compute the DW potential
    return coefA * c**2*(1 - c)**2