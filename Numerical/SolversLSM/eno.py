import numpy as np
# import math
# import matplotlib.pyplot as plt


def delta_plus(iphi, k):
    return iphi[k+1] - iphi[k]


def delta_minus(iphi, k):
    return iphi[k] - iphi[k-1]


# # # Create example
# x_min = -1
# x_max = 1
# nodes = 50
# dx = (x_max - x_min) / nodes
#
# x = np.linspace(x_min, x_max, nodes)
# phi = np.zeros(x.shape)
# dphi = np.zeros(x.shape)
#
# # for i in range(len(x)):
# #     phi[i] = abs(x[i]) - 1
#
# for i in range(len(x)):
#     phi[i] = (1/4) + (1/2) * math.sin(math.pi * x[i])
#     dphi[i] = (math.pi/2) * math.cos(math.pi * x[i])


def diff_bx(phi, dx):
    phi0_x = np.zeros(phi.shape)
    phi1_x = np.zeros(phi.shape)
    phi2_x = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        t1 = (1 * delta_plus(phi, n-3)) / (3 * dx)
        t2 = (-7 * delta_plus(phi, n-2)) / (6 * dx)
        t3 = (11 * delta_plus(phi, n-1)) / (6 * dx)

        phi0_x[n] = t1 + t2 + t3

        t1 = (-1 * delta_plus(phi, n-2)) / (6 * dx)
        t2 = (5 * delta_plus(phi, n-1)) / (6 * dx)
        t3 = (1 * delta_plus(phi, n+0)) / (3 * dx)

        phi1_x[n] = t1 + t2 + t3

        t1 = (1 * delta_plus(phi, n-1)) / (3 * dx)
        t2 = (5 * delta_plus(phi, n+0)) / (6 * dx)
        t3 = (-1 * delta_plus(phi, n+1)) / (6 * dx)

        phi2_x[n] = t1 + t2 + t3

    return phi0_x, phi1_x, phi2_x


def diff_fx(phi, dx):
    phi0_x = np.zeros(phi.shape)
    phi1_x = np.zeros(phi.shape)
    phi2_x = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1 = (1 * delta_minus(phi, n+3)) / (3 * dx)
        t2 = (-7 * delta_minus(phi, n+2)) / (6 * dx)
        t3 = (11 * delta_minus(phi, n+1))/ (6 * dx)

        phi0_x[n] = t1 + t2 + t3

        t1 = (-1 * delta_minus(phi, n+2)) / (6 * dx)
        t2 = (5 * delta_minus(phi, n+1)) / (6 * dx)
        t3 = (1 * delta_minus(phi, n+0)) / (3 * dx)

        phi1_x[n] = t1 + t2 + t3

        t1 = (1 * delta_minus(phi, n+1)) / (3 * dx)
        t2 = (5 * delta_minus(phi, n+0)) / (6 * dx)
        t3 = (-1 * delta_minus(phi, n-1)) / (6 * dx)

        phi2_x[n] = t1 + t2 + t3

    return phi0_x, phi1_x, phi2_x


def conditions(phi):
    c1 = np.zeros(phi.shape)
    c2 = np.zeros(phi.shape)
    c3 = np.zeros(phi.shape)
    c4 = np.zeros(phi.shape)
    c5 = np.zeros(phi.shape)
    c6 = np.zeros(phi.shape)

    'Condition 1'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i - 1)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i - 1)

    for i in range(3, len(phi) - 3):
        c1[i] = np.abs(phi_pp[i])

    'Condition 2'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i)

    for i in range(3, len(phi) - 3):
        c2[i] = np.abs(phi_pp[i])

    'Condition 3'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i - 1)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i - 1)

    for i in range(3, len(phi) - 3):
        phi_ppp[i] = delta_minus(phi_pp, i - 1)

    for i in range(3, len(phi) - 3):
        c3[i] = np.abs(phi_ppp[i])

    'Condition 4'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i - 1)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i - 1)

    for i in range(3, len(phi) - 3):
        phi_ppp[i] = delta_plus(phi_pp, i - 1)

    for i in range(3, len(phi) - 3):
        c4[i] = np.abs(phi_ppp[i])

    'Condition 5'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i)

    for i in range(3, len(phi) - 3):
        phi_ppp[i] = delta_minus(phi_pp, i)

    for i in range(3, len(phi) - 3):
        c5[i] = np.abs(phi_ppp[i])

    'Condition 6'
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for i in range(3, len(phi) - 3):
        phi_p[i] = delta_plus(phi, i)

    for i in range(3, len(phi) - 3):
        phi_pp[i] = delta_minus(phi_p, i)

    for i in range(3, len(phi) - 3):
        phi_ppp[i] = delta_plus(phi_pp, i)

    for i in range(3, len(phi) - 3):
        c6[i] = np.abs(phi_ppp[i])

    return c1, c2, c3, c4, c5, c6


def eno(phi, dx):

    padding = 3
    phi = np.pad(phi, padding, 'edge')

    'Backward for x'

    c1, c2, c3, c4, c5, c6 = conditions(phi)
    c = (c1 < c2) & (c3 < c4)
    cc = (c1 > c2) & (c5 > c6)
    phi0_x, phi1_x, phi2_x = diff_bx(phi, dx)
    dxb = np.where(c, phi0_x, (np.where(cc, phi2_x, phi1_x)))
    dxb = dxb[3:-3]

    'Forward for x'
    phi0_x, phi1_x, phi2_x = diff_fx(phi, dx)
    dxf = np.where(c, phi0_x, (np.where(cc, phi2_x, phi1_x)))
    dxf = dxf[3:-3]

    return dxb, dxf


# DxB, DxF = eno(phi, dx)
#
# plotting
# plt.grid(True)
# plt.axis([x_min, x_max, -2, 2])
# plt.plot(x, phi, 'black')
# plt.plot(x, dphi, 'green')
# plt.plot(x, DxB, 'red', linewidth=2.1)
# plt.plot(x, DxF, 'blue', linewidth=2.1)
# plt.show()



