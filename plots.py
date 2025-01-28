import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def PlotResults2D(folder:str, step: int, data_2d: np.array):
    # Create a figure and a set of subplots with a specified figure size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display an image on the axes, with a color map, origin, and value limits
    im = ax.imshow(data_2d, cmap='jet', origin="lower", vmin=0, vmax=1)

    # Add an axis for the color bar to the right of the main plot
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])

    # Create a color bar with the specified color axis
    cb = plt.colorbar(im, cax=cax)

    # Set the font size for the color bar ticks
    tick_font_size = 18
    cb.ax.tick_params(labelsize=tick_font_size)

    # Turn off the axis ticks and labels for the main plot
    ax.axis('off')

    # Save the figure to a file with the step number in the filename, at 300 dpi resolution and tight bounding box
    fig.savefig(f"{folder}/time_{str(step)}.png", dpi=300, bbox_inches='tight')

    # Close the figure
    plt.close(fig)

def PlotTotalEnergy(folder:str, energy:np.array):

    time = np.arange(0, len(energy), 1) # time data

    if np.max(energy) > 10000:
        n = 1000 # plotting scale
    else:
        n = 100

    if np.max(time) > 4000:
        t = 1000
    else:
        t = 100

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, energy, color="red")
    ax.set_ylim([np.round(np.min(energy), -2) - n, np.round(np.min(energy), -2) + n])
    ax.set_xlim([np.min(time) - 20, np.max(time) + 10])

    # Set major ticks
    ax.set_yticks(np.arange(np.round(np.min(energy), -2), np.round(np.max(energy), -2) + n+1, n))
    ax.set_xticks(np.arange(0, np.max(time) + t+1, t))

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
    ax.set_xlabel("Time step", fontsize=14)
    ax.set_ylabel("Total energy", fontsize=14)
    # ax.legend(fontsize=12, loc="upper left")

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(f"{folder}/energy_time.png", dpi=300)
    plt.sclose(fig)

def PlotResults3D(folder, step, data, N_shape, d_shape):
    Nx, Ny, Nz = N_shape
    dx, dy, dz = d_shape

    # Compute physical domain dimensions
    Lx, Ly, Lz = Nx * dx, Ny * dy, Nz * dz
    # Generate unstructured grid points
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    z = np.linspace(-Lz/2, Lz/2, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    kw = {
    'vmin': 0,
    'vmax': 1,
    'levels': np.linspace(0, 1.0, 10),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],  # Slice at Z=0
    zdir='z', offset=Z.max(), **kw
    )

    _ = ax.contourf(
        X[:, 0, :], data[:, 0, :], Z[:, 0, :],  # Slice at Y=0
        zdir='y', offset=Y.min(), **kw
    )

    C = ax.contourf(
        data[-1, :, :], Y[-1, :, :], Z[-1, :, :],  # Slice at X=max
        zdir='x', offset=X.max(), **kw
    )

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1.0, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='X [~]',
        ylabel='Y [~]',
        zlabel='Z [~]',
        # zticks=[0, -150, -300, -450],
    )

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    cbar = fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Concentration [~]')
    cbar.formatter = FormatStrFormatter('%.1f')  # Format tick labels to 1 decimal place
    cbar.update_ticks()

    fig.savefig(f"{folder}/time3D_{str(step)}.png", dpi=300, bbox_inches='tight')

    # Close the figure
    plt.close(fig)
