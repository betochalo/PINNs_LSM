import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchviz import make_dot
import graphviz


def graphs_loss(loss, lossphi, lossf):
    sns.set_style("whitegrid")
    colors = sns.color_palette('deep')

    def moving_average(data, window_size=10):
        """Función para calcular un promedio móvil."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(moving_average(loss, 50), label='Total Loss', color=colors[0])
    plt.legend()
    plt.title('Total Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')

    plt.subplot(1, 3, 2)
    plt.plot(moving_average(lossphi, 50), label='Loss_phi', color=colors[1])
    plt.legend()
    plt.title('Loss_phi')
    plt.xlabel('Iterations')

    plt.subplot(1, 3, 3)
    plt.plot(moving_average(lossf, 50), label='Loss_f', color=colors[2])
    plt.legend()
    plt.title('Loss_f')
    plt.xlabel('Iterations')

    plt.savefig('losses3.png', dpi=300)


def model_graph(pinns):
    layers = [3, 10, 10, 1]
    tf = np.random.uniform(low=0, high=1, size=(2, 1))
    xf = np.random.uniform(low=0, high=1, size=(2, 1))
    yf = np.random.uniform(low=0, high=1, size=(2, 1))

    t = np.random.uniform(low=0, high=1, size=(2, 1))
    x = np.random.uniform(low=0, high=1, size=(2, 1))
    y = np.random.uniform(low=0, high=1, size=(2, 1))

    u = np.random.uniform(low=0, high=1, size=(2, 1))
    v = np.random.uniform(low=0, high=1, size=(2, 1))

    phi = np.random.uniform(low=0, high=1, size=(2, 1))
    X_phi = np.hstack((x, y, t))
    vf = np.hstack((u, v))

    X_f = np.hstack((xf, yf, tf))

    model = pinns(X_phi=X_phi, X_f=X_f, vf=vf, phi=phi, layers=layers)
    x_sample = torch.randn(1, 3)
    output_sample = model.dnn(x_sample)
    make_dot(output_sample, params=dict(model.dnn.named_parameters())).render("nn", format='png')


# def compute_errors(phi_original, phi_approx):
#     assert phi_original.shape == phi_approx.shape
#
#     error = np.abs(phi_original - phi_approx)
#
#     l1_error = np.sum(error) / error.size
#     l2_error = np.sqrt(np.sum(error ** 2) / error.size)
#     inf_error = np.max(error)
#
#     return l1_error, l2_error, inf_error
#
#
# def heaviside(phi):
#     h = np.zeros(phi.shape)
#     (xl, yl) = phi.shape
#
#     for i in range(xl):
#         for j in range(yl):
#             if phi[i, j] < 0:
#                 h[i, j] = 1
#             else:
#                 h[i, j] = 0
#
#     return h
#
#
# def l1_error(h_expected, h_computed):
#     nodes = len(h_expected)
#     phi_error = np.zeros((nodes, nodes))
#
#     for i in range(nodes):
#         for j in range(nodes):
#             phi_error[i, j] = abs(h_expected[i, j] - h_computed[i, j])
#
#     l1 = np.sum(phi_error) / (nodes * nodes)
#
#     return l1, phi_error


def heaviside(phi):
    cond = phi <= 0

    h = np.where(cond, 1, 0)

    return h


def l1_error(phi_expected, phi_computed):
    phi1 = heaviside(phi_expected)
    phi2 = heaviside(phi_computed)

    error = np.abs(phi1 - phi2)

    l1 = np.sum(error) / error.size

    return l1


def l2_error(phi_expected, phi_computed):
    phi1 = heaviside(phi_expected)
    phi2 = heaviside(phi_computed)
    error = np.abs(phi1 - phi2)
    l2 = np.sqrt(np.sum(error ** 2) / error.size)
    return l2


def inf_error(phi_expected, phi_computed):
    phi1 = heaviside(phi_expected)
    phi2 = heaviside(phi_computed)
    error = np.abs(phi1 - phi2)
    l_inf = np.max(error)
    return l_inf


def area(phi):
    h = heaviside(phi)
    a = np.sum(h) / phi.size
    return a

# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# data = scipy.io.loadmat('./Data/data_M/LSM_1257x128x128.mat')
# phi_pred = data['phi']
#
# x_min = 0
# x_max = 1
# y_min = 0
# y_max = 1
# nodes = 128
#
# x, dx = np.linspace(x_min, x_max, nodes, retstep=True)
# y, dy = np.linspace(y_min, y_max, nodes, retstep=True)
#
# X, Y = np.meshgrid(x, y, indexing='ij')
#
# for i in range(0, 25):
#     plt.figure(figsize=(5, 5))
#     plt.contour(X, Y, phi_pred[50*i], levels=[0], colors='black')
#     plt.savefig('./Data/figures/' + '/phi_[i=%d].png' % (50*i))
#     plt.close('all')