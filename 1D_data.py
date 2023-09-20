# Libraries
import numpy as np
import scipy.io
import matplotlib
import copy
from Numerical.SolversLSM import eno
from Numerical.SolversLSM import runge_kutta3O as rk
import datetime
from matplotlib import pyplot as plt

# import math
matplotlib.rcParams['text.usetex'] = True

# Parameters
x_min = -2.0
x_max = 2.0
nodes = 320
dx = (x_max - x_min) / nodes

x = np.linspace(x_min, x_max, nodes)

# create function
phi = np.zeros(x.shape)
# Smooth function
for i in range(len(x)):
    phi[i] = abs(x[i]) - 1

phi_initial = copy.deepcopy(phi)
phi_initial = np.reshape(phi_initial, (nodes, 1))

tNow = 0
tMax = 50
dt = 0.1

u = 0.01 * np.ones(x.shape)

# # Create a data matrix mxn
alpha = dt * (np.max(np.abs(u) / dx))
data = []
t = np.array([])
# data = phi_initial
#####################################
counter = 0
phi_new = np.zeros(x.shape)
start = datetime.datetime.now()
while tNow < tMax:
    # Exact solution
    # Condiciones de borde
    phi[0] = abs(-2 - (0.01 * tNow)) - 1
    # phi[0] = 1
    # phi[nodes - 1] = 1

    dxb, dxf = eno.eno(phi, dx)
    phi_new = rk.tvd_runge_kutta_3o(phi, u, dxb, dxf, dx, dt)

    phi = copy.deepcopy(phi_new)
    # phi50 = copy.deepcopy(phi_new)
    phi1 = np.reshape(phi, (nodes, 1))
    phi_new = np.zeros(x.shape)
    t1 = np.array([tNow])
    if tNow == 0:
        t = np.concatenate((t, t1), axis=0)
    else:
        t = np.concatenate((t, t1), axis=0)
    if tNow == 0:
        data = phi1
    else:
        data = np.concatenate((data, phi1), axis=1)
    tNow = tNow + dt
    counter += 1

    plt.plot(x, phi, 'black')
    plt.grid(True)
    # plt.axis([x_min, x_max, -1.5, 1.5])

    if not (counter % 10):
        print(f't_now = {tNow}')
        plt.savefig(f'output/my_fig{counter:03d}.png')
        plt.clf()

end = datetime.datetime.now()
elapsed = end - start
print(f'Elapsed time: {elapsed}')

scipy.io.savemat('LSM2.mat', {'data': data, 'x': x, 'u': u, 't': t})

