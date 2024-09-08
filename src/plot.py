import os
import torch
import hydra

import numpy as np
import matplotlib.pyplot as plt


def plot_ad_potential(potential, traj_dihedral, start_dihedral, goal_dihedral, epoch):
    plt.clf()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sample_num = traj_dihedral[0].shape[0]
    traj_length = traj_dihedral[0].shape[1]

    # Plot the potential
    xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
    ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T
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
            traj_dihedral[0][idx],
            traj_dihedral[1][idx],
            marker="o",
            linestyle="None",
            markersize=4,
            alpha=1.0,
            zorder=101
        )

    # Plot start and goal states
    ax.scatter(
        start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=100, s=160
    )
    ax.scatter(
        goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=100, s=500, marker="*"
    )
    
    # Plot the Ramachandran plot
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.show()
    
    # Save the plot
    output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    if os.path.exists(f"{output_dir}/img") is False:
        os.mkdir(f"{output_dir}/img")
    img_path = f"{output_dir}/img/ram-{epoch}.png"
    plt.savefig(f"{img_path}")
    plt.close()
    
    return fig