import os
import torch
import hydra

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import colors


class AlaninePotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor(np.array([x, y]))
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad
    

class DoubleWellPotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        spacing = 68
        dims = [spacing, spacing]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // spacing, i % spacing, :] = torch.tensor(np.array([x, y]))
            self.data[i // spacing, i % spacing] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z


def save_plot(dir, name, fig):
    if os.path.exists(f"{dir}/img") is False:
        os.mkdir(f"{dir}/img")
    img_path = f"{dir}/img/{name}"
    fig.savefig(f"{img_path}")
    print(f"Saved plot at {img_path}")
    plt.close()


def plot_ad_potential(
    traj_dihedral,
    start_dihedral,
    goal_dihedral,
    cv_bound_use,
    cv_bound,
    epoch,
    name
):
    plt.clf()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    traj_list_phi = traj_dihedral[0]
    traj_list_psi = traj_dihedral[1]
    sample_num = len(traj_dihedral[0])

    # Plot the potential
    xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
    ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T
    potential = AlaninePotential(f"./data/alanine/final_frame.dat")
    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])
    plt.contourf(xs, ys, z, levels=100, zorder=0)

    # Plot the trajectory
    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / sample_num) for i in range(sample_num)]
    )
    for idx in range(sample_num):
        ax.plot(
            traj_list_phi[idx].cpu(),
            traj_list_psi[idx].cpu(),
            marker="o",
            linestyle="None",
            markersize=3,
            alpha=1.0,
            zorder=100
        )

    # Plot start and goal states
    ax.scatter(
        start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=101, s=160
    )
    ax.scatter(
        goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=101, s=500, marker="*"
    )
    if cv_bound_use and cv_bound > 0:
        square = plt.Rectangle(
            (goal_dihedral[0] - cv_bound / 2, goal_dihedral[1] - cv_bound /2),
            cv_bound, cv_bound,
            color='r', fill=False, linewidth=4,
            zorder=101
        )
        plt.gca().add_patch(square)
    
    # Plot the Ramachandran plot
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.show()
    
    save_plot(
        dir = hydra.core.hydra_config.HydraConfig.get().run.dir,
        name = f"{epoch}-ad-{name}.png",
        fig = fig
    )
    return fig

def plot_dw_potential(traj, start, goal, epoch):
    plt.clf()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sample_num = traj.shape[0]

    # Plot the potential
    bound = 1.7
    interval = 0.01
    xs = np.arange(-bound, bound, interval)
    ys = np.arange(-bound, bound, interval)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T
    potential = DoubleWellPotential(f"./data/double-well/potential.dat")
    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])
    plt.contourf(xs, ys, z, levels=100, zorder=0)

    # Plot the trajectory
    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / sample_num) for i in range(sample_num)]
    )
    for idx in range(sample_num):
        ax.plot(
            traj[idx, :, 0],
            traj[idx, :, 1],
            marker="o",
            linestyle="None",
            markersize=3,
            alpha=1.0,
            zorder=1
        )

    # Plot start, goal state, and square for cv bound
    ax.scatter(
        start[0], start[1], edgecolors="black", c="w", zorder=100, s=160
    )
    ax.scatter(
        goal[0], goal[1], edgecolors="black", c="w", zorder=100, s=500, marker="*"
    )
    square = plt.Rectangle((1.118 - 0.2, -0.2), 0.4, 0.4, color='r', fill=False, linewidth=2)
    plt.gca().add_patch(square)
    
    # Plot the Ramachandran plot
    plt.xlim([-bound, bound])
    plt.ylim([-bound, bound])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    
    # Save the plot
    output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    if os.path.exists(f"{output_dir}/img") is False:
        os.mkdir(f"{output_dir}/img")
    img_path = f"{output_dir}/img/dw-pot-{epoch}.png"
    plt.savefig(f"{img_path}")
    plt.close()
    
    return fig

def plot_ad_cv(
    phi: np.ndarray,
    psi: np.ndarray,
    cv: np.ndarray,
    cv_dim: int,
    epoch: int,
    start_dihedral,
    goal_dihedral,
    cfg_plot
):
    df = pd.DataFrame({
        **{f'CV{i}': cv[:, i] for i in range(cv_dim)},
        'psi': psi,
        'phi': phi
    })
    
    # Plot the projection of CVs
    fig, axs = plt.subplots(2, 2, figsize = ( 15, 12 ) )
    axs = axs.ravel()
    norm = colors.Normalize(
        vmin=min(df[f'CV{i}'].min() for i in range(min(cv_dim, 9))),
        vmax=max(df[f'CV{i}'].max() for i in range(min(cv_dim, 9)))
    )
    for i in range(min(cv_dim, 4)):
        ax = axs[i]
        df.plot.hexbin(
            'phi','psi', C=f"CV{i}",
            cmap=cfg_plot.cmap, ax=ax,
            gridsize=cfg_plot.gridsize,
            norm=norm
        )
        ax.scatter(start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=101, s=100)
        ax.scatter(goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=101, s=300, marker="*")
    
    save_plot(
        dir = hydra.core.hydra_config.HydraConfig.get().run.dir,
        name = f"{epoch}-ad-cv.png",
        fig = fig
    )
    fig_projection = fig
    
    # Plot the projection of CVs for meta-stable states
    fig, axs = plt.subplots(2, 2, figsize = ( 15, 12 ) )
    axs = axs.ravel()
    for i in range(min(cv_dim, 4)):
        ax = axs[i]
        df_filtered = df[
            (
                df.phi.between(left=start_dihedral[0].item()-0.5, right=start_dihedral[0].item()+0.5) &
                df.psi.between(left=start_dihedral[1].item()-0.5, right=start_dihedral[1].item()+0.5)
            ) |
            (
                df.phi.between(left=goal_dihedral[0].item()-0.5, right=goal_dihedral[0].item()+0.5) &
                df.psi.between(left=goal_dihedral[1].item()-0.5, right=goal_dihedral[1].item()+0.5)
            )
        ]
        df_filtered.plot.hexbin(
            'phi','psi', C=f"CV{i}",
            cmap=cfg_plot.cmap, ax=ax,
            gridsize=cfg_plot.gridsize
        )
        ax.scatter(start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=101, s=100)
        ax.scatter(goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=101, s=300, marker="*")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
    save_plot(
        dir = hydra.core.hydra_config.HydraConfig.get().run.dir,
        name = f"{epoch}-ad-cv-state.png",
        fig = fig
    )
    fig_state = fig
    
    # Plot the contour plot of CV of interest
    cv_index = cfg_plot.cv_index
    fig, axs = plt.subplots(3, 3, figsize = ( 15, 12 ) )
    axs = axs.ravel()
    max_cv0 = df[f'CV{cv_index}'].max()
    min_cv0 = df[f'CV{cv_index}'].min()
    boundary = np.linspace(min_cv0, max_cv0, cfg_plot.number_of_bins)
    
    for i in range(0, min(len(boundary)-1, 9)):
        ax = axs[i]
        df_selected = df[(df[f'CV{cv_index}'] <= boundary[i+1]) & (df[f'CV{cv_index}'] >= boundary[i])]
        if len(df_selected) == 0:
            continue
        df_selected.plot.hexbin(
            'phi', 'psi', C=f'CV{cv_index}',
            vmin = boundary[0], vmax = boundary[-1],
            cmap = cfg_plot.cmap, ax = ax,
            gridsize=cfg_plot.gridsize,
        )
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
    plt.tight_layout()
    
    save_plot(
        dir = hydra.core.hydra_config.HydraConfig.get().run.dir,
        name = f"{epoch}-ad-cv{cv_index}-div.png",
        fig = fig
    )
    fig_contur = fig
    
    return fig_projection, fig_state, fig_contur