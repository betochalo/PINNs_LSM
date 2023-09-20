import numpy as np
from Numerical.SolversLSM import eno as eno


def rk(phi, u, dxb, dxf, dt):
    cx1 = u < 0

    dx = np.where(cx1, dxf, dxb)

    space_diff = u * dx
    phi_new = phi - dt * space_diff

    return phi_new


def tvd_runge_kutta_3o(phi, u, dxb, dxf, dx, dt):
    phi1 = rk(phi, u, dxb, dxf, dt)
    dxb1, dxf1 = eno.eno(phi1, dx)
    phi2 = rk(phi1, u, dxb1, dxf1, dt)
    phi12 = (3/4) * phi + (1/4) * phi2
    dxb2, dxf2 = eno.eno(phi12, dx)
    phi32 = rk(phi12, u, dxb2, dxf2, dt)
    phi_new = (1/3) * phi + (2/3) * phi32

    return phi_new


