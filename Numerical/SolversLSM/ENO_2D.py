import numpy as np

# Differences

def delta_plus_x(phi, k):
    return phi[k + 1, 3:-3] - phi[k, 3:-3]


def delta_minus_x(phi, k):
    return phi[k, 3:-3] - phi[k - 1, 3:-3]


def delta_plus_y(phi, l):
    return phi[3:-3, l + 1] - phi[3:-3, l]


def delta_minus_y(phi, l):
    return phi[3:-3, l] - phi[3:-3, l - 1]


def diff_bx(phi, dx):
    phi0_x = np.zeros(phi.shape)
    phi1_x = np.zeros(phi.shape)
    phi2_x = np.zeros(phi.shape)

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (1 * delta_plus_x(phi, n - 3) / (3 * dx))
        t2[n, 3:-3] = (-7 * delta_plus_x(phi, n - 2) / (6 * dx))
        t3[n, 3:-3] = (11 * delta_plus_x(phi, n - 1) / (6 * dx))

    phi0_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (-1 * delta_plus_x(phi, n - 2) / (6 * dx))
        t2[n, 3:-3] = (5 * delta_plus_x(phi, n - 1) / (6 * dx))
        t3[n, 3:-3] = (1 * delta_plus_x(phi, n + 0) / (3 * dx))

    phi1_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (1 * delta_plus_x(phi, n - 1) / (3 * dx))
        t2[n, 3:-3] = (5 * delta_plus_x(phi, n + 0) / (6 * dx))
        t3[n, 3:-3] = (-1 * delta_plus_x(phi, n + 1) / (6 * dx))

    phi2_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    return phi0_x, phi1_x, phi2_x


def diff_fx(phi, dx):
    phi0_x = np.zeros(phi.shape)
    phi1_x = np.zeros(phi.shape)
    phi2_x = np.zeros(phi.shape)

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (1 * delta_minus_x(phi, n + 3) / (3 * dx))
        t2[n, 3:-3] = (-7 * delta_minus_x(phi, n + 2) / (6 * dx))
        t3[n, 3:-3] = (11 * delta_minus_x(phi, n + 1) / (6 * dx))

    phi0_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (-1 * delta_minus_x(phi, n + 2) / (6 * dx))
        t2[n, 3:-3] = (5 * delta_minus_x(phi, n + 1) / (6 * dx))
        t3[n, 3:-3] = (1 * delta_minus_x(phi, n + 0) / (3 * dx))

    phi1_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[n, 3:-3] = (1 * delta_minus_x(phi, n + 1) / (3 * dx))
        t2[n, 3:-3] = (5 * delta_minus_x(phi, n + 0) / (6 * dx))
        t3[n, 3:-3] = (-1 * delta_minus_x(phi, n - 1) / (6 * dx))

    phi2_x[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    return phi0_x, phi1_x, phi2_x


def diff_by(phi, dy):
    phi0_y = np.zeros(phi.shape)
    phi1_y = np.zeros(phi.shape)
    phi2_y = np.zeros(phi.shape)

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (1 * delta_plus_y(phi, n - 3) / (3 * dy))
        t2[3:-3, n] = (-7 * delta_plus_y(phi, n - 2) / (6 * dy))
        t3[3:-3, n] = (11 * delta_plus_y(phi, n - 1) / (6 * dy))

    phi0_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (-1 * delta_plus_y(phi, n - 2) / (6 * dy))
        t2[3:-3, n] = (5 * delta_plus_y(phi, n - 1) / (6 * dy))
        t3[3:-3, n] = (1 * delta_plus_y(phi, n + 0) / (3 * dy))

    phi1_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (1 * delta_plus_y(phi, n - 1) / (3 * dy))
        t2[3:-3, n] = (5 * delta_plus_y(phi, n + 0) / (6 * dy))
        t3[3:-3, n] = (-1 * delta_plus_y(phi, n + 1) / (6 * dy))

    phi2_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    return phi0_y, phi1_y, phi2_y


def diff_fy(phi, dy):
    phi0_y = np.zeros(phi.shape)
    phi1_y = np.zeros(phi.shape)
    phi2_y = np.zeros(phi.shape)

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (1 * delta_minus_y(phi, n + 3) / (3 * dy))
        t2[3:-3, n] = (-7 * delta_minus_y(phi, n + 2) / (6 * dy))
        t3[3:-3, n] = (11 * delta_minus_y(phi, n + 1) / (6 * dy))

    phi0_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (-1 * delta_minus_y(phi, n + 2) / (6 * dy))
        t2[3:-3, n] = (5 * delta_minus_y(phi, n + 1) / (6 * dy))
        t3[3:-3, n] = (1 * delta_minus_y(phi, n + 0) / (3 * dy))

    phi1_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    t1 = np.zeros(phi.shape)
    t2 = np.zeros(phi.shape)
    t3 = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        t1[3:-3, n] = (1 * delta_minus_y(phi, n + 1) / (3 * dy))
        t2[3:-3, n] = (5 * delta_minus_y(phi, n + 0) / (6 * dy))
        t3[3:-3, n] = (-1 * delta_minus_y(phi, n - 1) / (6 * dy))

    phi2_y[3:-3, 3:-3] = t1[3:-3, 3:-3] + t2[3:-3, 3:-3] + t3[3:-3, 3:-3]

    return phi0_y, phi1_y, phi2_y


# Compute conditions
def conditions_x(phi):
    cx1 = np.zeros(phi.shape)
    cx2 = np.zeros(phi.shape)
    cx3 = np.zeros(phi.shape)
    cx4 = np.zeros(phi.shape)
    cx5 = np.zeros(phi.shape)
    cx6 = np.zeros(phi.shape)

    # condition 1 for  x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n - 1)

    cx1[3:-3, 3:-3] = np.abs(phi_pp[3:-3, 3:-3])

    # condition 2 for x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n + 0)

    cx2[3:-3, 3:-3] = np.abs(phi_pp[3:-3, 3:-3])

    # condition 3 for x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n - 1)

    for n in range(3, len(phi) - 3):
        phi_ppp[n, 3:-3] = delta_minus_x(phi_pp, n - 1)

    cx3[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 4 for x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n - 1)

    for n in range(3, len(phi) - 3):
        phi_ppp[n, 3:-3] = delta_plus_x(phi_pp, n - 1)

    cx4[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 5 for x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n + 0)

    for n in range(3, len(phi) - 3):
        phi_ppp[n, 3:-3] = delta_minus_x(phi_pp, n + 0)

    cx5[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 6 for x
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[n, 3:-3] = delta_plus_x(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[n, 3:-3] = delta_minus_x(phi_p, n + 0)

    for n in range(3, len(phi) - 3):
        phi_ppp[n, 3:-3] = delta_plus_x(phi_pp, n + 0)

    cx6[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    return cx1, cx2, cx3, cx4, cx5, cx6


def conditions_y(phi):
    cy1 = np.zeros(phi.shape)
    cy2 = np.zeros(phi.shape)
    cy3 = np.zeros(phi.shape)
    cy4 = np.zeros(phi.shape)
    cy5 = np.zeros(phi.shape)
    cy6 = np.zeros(phi.shape)

    # condition 1 para y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n - 1)

    cy1[3:-3, 3:-3] = np.abs(phi_pp[3:-3, 3:-3])

    # condition 2 for y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n + 0)

    cy2[3:-3, 3:-3] = np.abs(phi_pp[3:-3, 3:-3])

    # condition 3 for y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n - 1)

    for n in range(3, len(phi) - 3):
        phi_ppp[3:-3, n] = delta_minus_y(phi_pp, n - 1)

    cy3[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 4 for y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n - 1)

    for n in range(3, len(phi) - 3):
        phi_ppp[3:-3, n] = delta_plus_y(phi_pp, n - 1)

    cy4[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 5 for y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n + 0)

    for n in range(3, len(phi) - 3):
        phi_ppp[3:-3, n] = delta_minus_y(phi_pp, n + 0)

    cy5[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    # condition 6 for y
    phi_p = np.zeros(phi.shape)
    phi_pp = np.zeros(phi_p.shape)
    phi_ppp = np.zeros(phi_p.shape)

    for n in range(3, len(phi) - 3):
        phi_p[3:-3, n] = delta_plus_y(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_pp[3:-3, n] = delta_minus_y(phi_p, n + 0)

    for n in range(3, len(phi) - 3):
        phi_ppp[3:-3, n] = delta_plus_y(phi_pp, n + 0)

    cy6[3:-3, 3:-3] = np.abs(phi_ppp[3:-3, 3:-3])

    return cy1, cy2, cy3, cy4, cy5, cy6


def eno(phi, dx, dy):
    """
    This function implements the ideas of the following papers:
    [1] G.-S. Jiang and D. Peng, “Weighted ENO Schemes for Hamilton--Jacobi Equations,” SIAM J. Sci. Comput., vol. 21,
        no. 6, pp. 2126–2143, Jan. 2000.
    [2] I. Pineda, D. Arellano, and R. Chachalo, “Analysis of Essentially Non-Oscillatory Numerical Techniques for the
        Computation of the Level Set Method,” in International Conference on Applied Technologies, 2019, vol. 0, no. 2.
    :param phi:
    :param dx:
    :param dy:
    :param nodes:
    :return:
    """
    padding = 3
    phi = np.pad(phi, padding, 'reflect', reflect_type='odd')

    # Backward for x
    cx1, cx2, cx3, cx4, cx5, cx6 = conditions_x(phi)
    c = (cx1 < cx2) & (cx3 < cx4)
    c1 = (cx1 > cx2) & (cx5 > cx6)
    phi0_x, phi1_x, phi2_x = diff_bx(phi, dx)
    dxb = np.where(c, phi0_x, (np.where(c1, phi2_x, phi1_x)))
    dxb = dxb[3:-3, 3:-3]

    # Forward for x
    phi0_x, phi1_x, phi2_x = diff_fx(phi, dx)
    dxf = np.where(c, phi0_x, (np.where(c1, phi2_x, phi1_x)))
    dxf = dxf[3:-3, 3:-3]

    # Backward for y
    cy1, cy2, cy3, cy4, cy5, cy6 = conditions_y(phi)
    c2 = (cy1 < cy2) & (cy3 < cy4)
    c3 = (cy1 > cy2) & (cy5 > cy6)
    phi0_y, phi1_y, phi2_y = diff_by(phi, dy)
    dyb = np.where(c2, phi0_y, (np.where(c3, phi2_y, phi1_y)))
    dyb = dyb[3:-3, 3:-3]

    # Forward for y
    phi0_y, phi1_y, phi2_y = diff_fy(phi, dy)
    dyf = np.where(c2, phi0_y, (np.where(c3, phi2_y, phi1_y)))
    dyf = dyf[3:-3, 3:-3]

    return dxb, dxf, dyb, dyf
