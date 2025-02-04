import torch
from matplotlib import animation
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy import ndimage
from copy import deepcopy
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import numpy as np

def diverging_alpha(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        cmap.colors = deepcopy(cmap.colors)
        for i, a in enumerate(cmap.colors):
            a.append(2 * abs(i / cmap.N - 0.5))
    elif isinstance(cmap, LinearSegmentedColormap):
        cmap._segmentdata["alpha"] = np.array(
            [[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )
    return cmap

def plot_3d_rows(u, t=[1.0], path=None):
    from vape4d import render
    # u in shape n_steps b t d h w c
    colormap = diverging_alpha(plt.get_cmap("magma"))
    u = u[:, 0, :, :, :, :, :3]  # n_steps nt nz nx ny 3 take only velocity components
    u = torch.norm(u, dim=-1) # n_steps nt nz nx ny

    u = torch.flip(u, dims=[2, 4]) # orient in 3D space correctly just for plotting
    u = torch.rot90(u, k=1, dims=[3, 4])

    fig, axs = plt.subplots(u.shape[0], len(t), figsize=(6*len(t), 6*u.shape[0]))

    for i in range(u.shape[0]):
        for j in range(len(t)):
            img = render(
                u[i],
                colormap,
                t[j], # timestep
                width=512,
                height=512,
            )

            axs[i][j].imshow(img)
            axs[i][j].axis("off")
            axs[i][j].title.set_text(f'Step: {j}, Timestep: {t[j]}')
    
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

    fig, axs = plt.subplots(u.shape[0], len(t), figsize = (12*u.shape[0], 12*len(t)))

    tmax = u.shape[1] - 1
    
    for i in range(u.shape[0]):
        for j in range(len(t)):
            t_plot = int(t[j]*tmax)
            u_slice = u[i, t_plot, :, :, 12]
            im = axs[i][j].imshow(ndimage.rotate(u_slice, 90), cmap='viridis')
            axs[i][j].title.set_text(f'Step: {i}, Timestep: {t[j]}')
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])

    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(path.replace(".png", "_slice.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

def plot_3d_batch(u, t=[1.0], path=None):
    from vape4d import render
    # u in shape b t d h w c
    colormap = diverging_alpha(plt.get_cmap("magma"))
    u = u[0, :, :, :, :, :3]  # nt nz nx ny 3 take only velocity components
    u = torch.norm(u, dim=-1) # nt nz nx ny

    u = torch.flip(u, dims=[1, 3]) # orient in 3D space correctly just for plotting
    u = torch.rot90(u, k=1, dims=[2, 3])

    fig, axs = plt.subplots(1, len(t), figsize=(6*len(t), 6))

    for j in range(len(t)):
        img = render(
            u,
            colormap,
            t[j], # timestep
            width=512,
            height=512,
        )

        axs[j].imshow(img)
        axs[j].axis("off")
        axs[j].title.set_text(f'Timestep: {t[j]}')
    
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

    fig, axs = plt.subplots(1, len(t), figsize = (12*len(t), 12))

    tmax = u.shape[0] - 1
    
    for j in range(len(t)):
        t_plot = int(t[j]*tmax)
        u_slice = u[t_plot, :, :, 12]
        im = axs[j].imshow(ndimage.rotate(u_slice, 90), cmap='viridis')
        axs[j].title.set_text(f'Step: {j}, Timestep: {t[j]}')
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])

    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(path.replace(".png", "_slice.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

def plot_3d(u, path, t=1.0):
    from vape4d import render
    # u in shape t d h w
    colormap = diverging_alpha(plt.get_cmap("magma"))
    img = render(
            # [T,D,H,W]
            torch.flip(u, dims=[1, 3]),
            colormap,
            t, # timestep
            width=1024,
            height=1024,
        )

    plt.imshow(img)
    plt.axis("off")

    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close() 

    tmax = u.shape[0] - 1
    t_plot = int(t*tmax)
    u_slice = u[t_plot, :, :, 12] # get the slice at the middle of the domain

    plt.imshow(u_slice, cmap='viridis')
    plt.colorbar(orientation='horizontal')
    plt.savefig(path.replace(".png", "_slice.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

def plot_mesh_batch(u, mesh_pos, cells, n_t, path=None):
    # u in shape n_diff, nt m
    # mesh_pos in shape m, 2
    # cells in shape n_edges, 3
    # plots time-dependent mesh data at n_t timesteps

    batch = u.shape[0]

    n_skip = u.shape[1] // n_t 
    u_downs = u[:, ::n_skip]

    fig, ax = plt.subplots(n_t, batch, figsize=(6*batch, 2*n_t))

    for j in range(batch):
        vmin = torch.min(u[j])
        vmax = torch.max(u[j])
        for i in range(n_t):
            ax[i][j].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = u_downs[j, i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i][j].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i][j].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i][j].title.set_text(f'Batch: {j}, Timestep: {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    fig.colorbar(tpc, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_grid_batch(u, n_t, path=None):
    # u in shape n_steps, nt, nx, ny
    # plots time-dependent mesh data at n_t timesteps

    batch = u.shape[0]

    n_skip = u.shape[1] // n_t 
    u_downs = u[:, ::n_skip]

    fig, ax = plt.subplots(n_t, batch, figsize=(4*batch, 4*n_t))

    for j in range(batch):
        vmin = torch.min(u[j])
        vmax = torch.max(u[j])
        for i in range(n_t):
            ax[i][j].set_axis_off()
            velocity = u_downs[j, i]
            im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
            ax[i][j].title.set_text(f'Batch: {j}, Timestep: {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)


def plot_grid(u, rec=None, n_t=1, path=None, flip=False):
    # u in shape nt nx ny
    vmin = torch.min(u)
    vmax = torch.max(u)

    n_skip = u.shape[0] // n_t 
    u_downs = u[::n_skip]

    if flip: 
        u_downs = ndimage.rotate(u_downs, 90, axes=(1, 2))

    if rec is not None:
        rec_downs = rec[::n_skip]
        if flip:
            rec_downs = ndimage.rotate(rec_downs, 90, axes=(1, 2))
        fig, ax = plt.subplots(n_t, 2, figsize=(8, 4*n_t))
        for j in range(2):
            for i in range(n_t):
                ax[i][j].set_axis_off()
                if j == 0:
                    velocity = u_downs[i] 
                else:
                    velocity = rec_downs[i]

                im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
                ax[i][j].title.set_text(f'Timestep {i*n_skip}')
            ax[0][j].title.set_text(f'Ground Truth' if j == 0 else f'Reconstruction')
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))

        for i in range(n_t):
            ax[i].set_axis_off()
            velocity = u_downs[i] 

            im = ax[i].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
            ax[i].title.set_text(f'Timestep {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_mesh(u, mesh_pos, cells, n_t, rec = None, path=None):
    # u in shape nt m
    # mesh_pos in shape m, 2
    # cells in shape n_edges, 3
    # plots time-dependent mesh data at n_t timesteps

    vmin = torch.min(u)
    vmax = torch.max(u)

    n_skip = u.shape[0] // n_t 
    u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 2, figsize=(12, 2*n_t))
        rec_downs = rec[::n_skip]
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(6, 2*n_t))

    if rec is None:
        for i in range(n_t):
            ax[i].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = u_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i].title.set_text(f'Timestep {i*n_skip}')
    else:
        for i in range(n_t):
            ax[i][0].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = u_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i][0].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i][0].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i][0].title.set_text(f'Ground Truth')

            ax[i][1].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = rec_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i][1].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i][1].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i][1].title.set_text(f'Pred')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(tpc, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)


def animate_mesh(u, u_hat, mesh_pos, cells, path):
    # u in shape nt m
    # mesh_pos in shape m, 2
    # cells in shape n_edges, 3

    t = u.shape[0]

    vmin = torch.min(u)
    vmax = torch.max(u)

    fig, ax = plt.subplots(2, 1, figsize=(24, 8))

    def animate(t):
        step = t
        ax[0].cla()
        ax[0].set_axis_off()

        pos = mesh_pos
        faces = cells
        velocity = u[step]
        triang0 = mtri.Triangulation(pos[:,0], pos[:,1], faces)
        tpc = ax[0].tripcolor(triang0, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[0].triplot(triang0, 'ko-', ms=0.5, lw=0.3)
        ax[0].title.set_text('Ground Truth')

        ax[1].cla()
        ax[1].set_axis_off()

        pos = mesh_pos
        faces = cells
        velocity = u_hat[step]
        triang1 = mtri.Triangulation(pos[:,0], pos[:,1], faces)
        ax[1].tripcolor(triang1, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[1].triplot(triang1, 'ko-', ms=0.5, lw=0.3)
        ax[1].title.set_text('Reconstruction')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(tpc, cax=cbar_ax)

        return fig,

    ani = animation.FuncAnimation(fig, animate, frames=t, interval=250)
    ani.save(path, writer='ffmpeg', fps=15)
    plt.close(fig)

def animate_grid(u, u_hat=None, path=None, upsample=1, flip=False):
    # u in shape nt nx ny, u_hat in shape nt nx ny

    if upsample > 1:
        u = F.interpolate(u.permute(1, 2, 0), scale_factor=upsample, mode="linear").permute(2, 0, 1)
        if u_hat is not None:
            u_hat = F.interpolate(u_hat.permute(1, 2, 0), scale_factor=upsample, mode="linear").permute(2, 0, 1)

    vmin = torch.min(u)
    vmax = torch.max(u)
    nt = u.shape[0]

    if u_hat is not None:
        fig, axs = plt.subplots(2, 1, figsize=(4, 8))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    if flip:
        u = ndimage.rotate(u, 90, axes=(1, 2))
        if u_hat is not None:
            u_hat = ndimage.rotate(u_hat, 90, axes=(1, 2))

    def animate(t):
        if u_hat is not None:
            im0 = axs[0].imshow(u[t], 
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap='inferno', 
                    aspect='equal')
            
            axs[0].title.set_text('Ground Truth')

            im1 = axs[1].imshow(u_hat[t],
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap='inferno', 
                    aspect='equal')
            axs[1].title.set_text('Reconstruction')

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        else:
            im0 = axs.imshow(u[t], 
                    vmin=vmin, 
                    vmax=vmax, 
                    cmap='inferno', 
                    aspect='equal')
            axs.title.set_text('Ground Truth')
                
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im0, cax=cbar_ax)

        return fig,

    ani = animation.FuncAnimation(fig, animate, frames=nt, interval=100)
    ani.save(path, writer='ffmpeg', fps=10)
    plt.close(fig)