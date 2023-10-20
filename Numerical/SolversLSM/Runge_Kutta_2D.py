import numpy as np
from Numerical.SolversLSM import ENO_2D as eno2d


def steps(phi, u, v, dxb, dxf, dyb, dyf, dt):
    cx = u < 0
    cy = v < 0

    dx = np.where(cx, dxf, dxb)
    dy = np.where(cy, dyf, dyb)

    space_diff = u * dx + v * dy
    phi_new = phi - dt * space_diff

    return phi_new


def tvd_runge_kutta_2d_3o(phi, u, v, dxb, dxf, dyb, dyf, dx, dy, dt):
    phi1 = steps(phi, u, v, dxb, dxf, dyb, dyf, dt)
    dxb1, dxf1, dyb1, dyf1 = eno2d.eno(phi1, dx, dy)
    phi2 = steps(phi1, u, v, dxb1, dxf1, dyb1, dyf1, dt)
    phi12 = (3 / 4) * phi + (1 / 4) * phi2
    dxb2, dxf2, dyb2, dyf2 = eno2d.eno(phi12, dx, dy)
    phi32 = steps(phi12, u, v, dxb2, dxf2, dyb2, dyf2, dt)
    phi_new = (1 / 3) * phi + (2 / 3) * phi32

    return phi_new
