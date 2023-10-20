import numpy as np
import math
import scipy.io
# import matplotlib
import copy
from Numerical.SolversLSM import ENO_2D as eno2
from Numerical.SolversLSM import Runge_Kutta_2D as rk2
from Numerical.SolversLSM import LSM_functions as lfun
import datetime
from matplotlib import pyplot as plt

# Parameters
x_min = 0
x_max = 1
y_min = 0
y_max = 1
nodes = 20
A = np.array([0.50, 0.75])
B = np.array([0.34, 0.65])
C = np.array([0.34, 0.85])


x, dx = np.linspace(x_min, x_max, nodes, retstep=True)
y, dy = np.linspace(y_min, y_max, nodes, retstep=True)

X, Y = np.meshgrid(x, y, indexing='ij')
# X, Y = np.meshgrid(x, y)
# phi = np.tanh(lfun.packman(X, Y, A, B, C, 0.50, 0.75, 0.15, 0.5, 0.85, 0.02))
phi = lfun.rectangle(X, Y, 0.5, 0.5, 0.10, 0.10)
# phi = lfun.circle(X, Y, 0.75, 0.50, 0.15)
# phi = lfun.zalesak_disk(X, Y, 0.5, 0.75, 0.15, 0.5, 0.60, 0.10, 0.25)
# phi = lfun.nut(X, Y, 0.5, 0.75, 0.15, 0.5, 0.6, 0.10, 0.25, 0.65, 0.75, 0.25, 0.10)
phi1 = copy.deepcopy(phi)
# Setting vector field
u = (math.pi / 314) * (.5 - Y)
v = (math.pi / 314) * (X - .5)
# u = (np.sin(math.pi*X))**2 * np.sin(2*math.pi*Y)
# v = - (np.sin(math.pi*Y))**2 * np.sin(2*math.pi*X)
plt.figure(figsize=(5, 5))
# plt.contourf(X, Y, phi, levels=[-1e6, 0], colors='green')
# plt.contour(X, Y, phi, levels=[0], colors='black')
plt.quiver(X, Y, u, v)
plt.savefig('img/circlevf.png', dpi=600)
plt.show()


# # Evolution
# tNow = 0
# tMax = 200
# dt = 0.1  #deformation 0.001
# sol = []
# u1 = []
# v1 = []
# t = np.array([])
# counter = 0
# # phi_new = np.zeros((len(X), len(Y)))
# start = datetime.datetime.now()
#
# while tNow < tMax:
#     # sol.append(phi)
#     # u1.append(u)
#     # v1.append(v)
#     dxb, dxf, dyb, dyf = eno2.eno(phi, dx, dy)
#     phi_new = rk2.tvd_runge_kutta_2d_3o(phi, u, v, dxb, dxf, dyb, dyf, dx, dy, dt)
#
#     phi = copy.deepcopy(phi_new)
#     # t1 = np.array([tNow])
#     # if tNow == 0:
#     #     t = np.concatenate((t, t1), axis=0)
#     # else:
#     #     t = np.concatenate((t, t1), axis=0)
#
#     tNow = tNow + dt
#     counter += 1
#     # print(f't_now = {tNow}')
#     # plt.figure(figsize=(5, 5))
#     plt.contourf(X, Y, phi, levels=[-1e6, 0], colors='yellow')
#     plt.contour(X, Y, phi, levels=[0], colors='black')
#     plt.quiver(X, Y, u, v)
#
#     if not (counter % 10):
#         print(f't_now = {tNow}')
#         plt.savefig(f'output3/my_fig{counter:03}.png')
#         plt.clf()
# end = datetime.datetime.now()
# elapsed = end - start
# print(f'Elapsed time: {elapsed}')
# # Crear un arreglo 3D
# data = None
# u2 = None
# v2 = None
# for solution in sol:
#     if data is None:
#         data = solution[np.newaxis, :, :]
#     else:
#         data = np.concatenate((data, solution[np.newaxis, :, :]), axis=0)
#
# for i in u1:
#     if u2 is None:
#         u2 = i[np.newaxis, :, :]
#     else:
#         u2 = np.concatenate((u2, i[np.newaxis, :, :]), axis=0)
#
# for i in v1:
#     if v2 is None:
#         v2 = i[np.newaxis, :, :]
#     else:
#         v2 = np.concatenate((v2, i[np.newaxis, :, :]), axis=0)
#
# # Save data in .mat
# scipy.io.savemat('LSM_2D_CIRCLE_deformation.mat', {'phi_sol': data, 'x': x, 'y': y, 'u': u2, 'v': v2, 't': t})
