import matplotlib.pyplot as plt
import numpy as np
from .utils import *
device = set_gpu(True)

def visualize_2d_data(train_data, test_data, train_labels=None, test_labels=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    plt.show()


def show_2d_latents(latents, labels, title='Latent Space'):
    plt.figure()
    plt.title(title)
    plt.scatter(latents[:, 0], latents[:, 1], s=1, c=labels)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.show()


def show_2d_densities(densities, x_lim, y_lim, title='Densities'):
    plt.figure()
    plt.title(title)
    dx, dy = 0.025, 0.025
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.show()

def show_2d_heatmap(fun, device, r=None, x_lim=(-2.4, 2.4), y_lim=(-1.5, 1.5), title="Densities"):
    dx, dy = 0.025, 0.025
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2)).to(device)

    if r == None:
      densities = np.exp(get_numpy(fun(mesh_xs)))
    else:
      dims=y.shape[0]*y.shape[1]
      u=(torch.zeros(dims, 1)+r).to(device)
      densities = np.exp(get_numpy(fun(mesh_xs, u)))
    show_2d_densities(densities, x_lim, y_lim, title)
