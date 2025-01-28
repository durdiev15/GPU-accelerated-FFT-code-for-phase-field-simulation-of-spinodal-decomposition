import numpy as np
import matplotlib.pyplot as plt

def fourier_frequency(Nx, dx, Ny, dy):
    kx = 2.0*np.pi*np.fft.fftfreq(Nx, dx)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny, dy)        
    kx_grid, ky_grid = np.meshgrid(kx,ky, indexing='ij')
    return np.array([kx_grid, ky_grid])

def plot_results(step, data):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(data, cmap='jet', origin="lower", vmin=0, vmax=1)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cb = plt.colorbar(im,cax=cax)
    tick_font_size = 18
    cb.ax.tick_params(labelsize=tick_font_size)
    ax.axis('off')
    fig.savefig("time_" + str(step), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def gradient_energy(c_k, freq, k_grad):
    
    # Compute gradients in Fourier space
    dc_dx_hat = 1j * freq[0] * c_k  # ∂c/∂x in Fourier space
    dc_dy_hat = 1j * freq[1] * c_k  # ∂c/∂y in Fourier space
    
    # Transform gradients back to real space
    dc_dx = np.fft.ifft2(dc_dx_hat).real
    dc_dy = np.fft.ifft2(dc_dy_hat).real
    
    # Compute the gradient energy
    return 0.5*k_grad * (dc_dx**2 + dc_dy**2)

from matplotlib.ticker import AutoMinorLocator
def plot_energy(time, energy):
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, energy*(Nx*Ny))
    # plt.xlabel("Time")
    # plt.ylabel("Total Energy")
    # plt.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("energy_time.png", dpi=300)
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, energy, color="red")
    ax.set_ylim([300, 1000])
    ax.set_xlim([-100, 8200])

    # Set major ticks
    ax.set_yticks(np.arange(300, 901, 100))
    ax.set_xticks(np.arange(0, 8001, 1000))

    # Set minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 minor ticks per major tick
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Customize tick lengths and font sizes
    ax.tick_params(axis='x', which='major', length=10, labelsize=12)
    ax.tick_params(axis='x', which='minor', length=5)
    ax.tick_params(axis='y', which='major', length=10, labelsize=12)
    ax.tick_params(axis='y', which='minor', length=5)

    # Add a grid for better visualization
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    # Add labels and legend
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Total Energy", fontsize=14)
    # ax.legend(fontsize=12, loc="upper left")

    # Save and display the plot
    plt.tight_layout()
    plt.savefig("energy_time.png", dpi=300)
    plt.show()


def Cahn_Hilliard_2D(Nx=128, Ny=128, dx=1, dy=1, nsteps=8000, nt=1000, dt=5e-2, 
                              coefA=1.0, c0=0.4, mob=1.0, k_grad=0.5):
    freq = fourier_frequency(Nx, dx, Ny, dy) # Fourier frequencies
    k2 = freq[0]**2 + freq[1]**2
    k4 = k2**2
    c = c0 + 0.02 * (0.5 - np.random.rand(Ny,Ny)) # initial order parameter
    plot_results(0, c) # plot the initial data
    energy = np.zeros(nsteps)
    for step in range(nsteps):
        print("\n ------------------ "+str(step) + " ---------------------\n")
        fc = coefA * c**2*(1 - c)**2
        c_k = np.fft.fft2(c)
        fgrad=gradient_energy(c_k, freq, k_grad)
        energy[step] = np.mean(fc + fgrad)
        dfc_dc = coefA * (c**2*(2*c - 2) + 2*c*(1 - c)**2)
        dfc_dc_k = np.fft.fft2(dfc_dc)
        # --------------------------------------------------------------------
        c_k = (c_k - dt * k2 * mob * dfc_dc_k) / (1 + dt * k4 * mob * k_grad)
        c = np.fft.ifft2(c_k).real
        if ((step + 1) % nt == 0 ):
            plot_results(step + 1, c)
    time = np.arange(0, nsteps, 1)
    plot_energy(time, energy*(Nx*Ny))
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, energy*(Nx*Ny))
    # plt.xlabel("Time")
    # plt.ylabel("Total Energy")
    # plt.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("energy_time.png", dpi=300)

def main():
    Cahn_Hilliard_2D()

if __name__ == "__main__":
    main()


