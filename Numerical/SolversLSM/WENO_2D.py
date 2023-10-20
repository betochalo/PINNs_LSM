import numpy as np

# Differences


def delta_plus_x(phi, k):
    return phi[k+1, 3:-3] - phi[k, 3:-3]


def delta_minus_x(phi, k):
    return phi[k, 3:-3] - phi[k-1, 3:-3]


def delta_plus_y(phi, l):
    return phi[3:-3, l+1] - phi[3:-3, l]


def delta_minus_y(phi, l):
    return phi[3:-3, l] - phi[3:-3, l-1]


def parameters_bx(phi, dx):
    # Compute parameter a

    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n-2)

    for n in range(3, len(phi)-3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n-2)

    a = phi_f / dx

    # Compute parameter b
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n-1)

    for n in range(3, len(phi)-3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n-1)

    b = phi_f / dx

    # Compute parameter c
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n+0)

    for n in range(3, len(phi)-3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n+0)

    c = phi_f / dx

    # Compute parameter d
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n+1)

    for n in range(3, len(phi)-3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n+1)

    d = phi_f / dx

    return a, b, c, d


def parameters_fx(phi, dx):
    # Compute parameter a

    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n + 2)

    for n in range(3, len(phi) - 3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n + 2)

    a = phi_f / dx

    # Compute parameter b
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n + 1)

    for n in range(3, len(phi) - 3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n + 1)

    b = phi_f / dx

    # Compute parameter c
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n + 0)

    c = phi_f / dx

    # Compute parameter d
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[n, 3:-3] = delta_plus_x(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_f[n, 3:-3] = delta_minus_x(phi_b, n - 1)

    d = phi_f / dx

    return a, b, c, d


def parameters_by(phi, dy):
    # Compute parameter a

    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n - 2)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n - 2)

    a = phi_f / dy

    # Compute parameter b
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n - 1)

    b = phi_f / dy

    # Compute parameter c
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n + 0)

    c = phi_f / dy

    # Compute parameter d
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n + 1)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n + 1)

    d = phi_f / dy

    return a, b, c, d


def parameters_fy(phi, dy):
    # Compute parameter a

    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n + 2)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n + 2)

    a = phi_f / dy

    # Compute parameter b
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n + 1)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n + 1)

    b = phi_f / dy

    # Compute parameter c
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n + 0)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n + 0)

    c = phi_f / dy

    # Compute parameter d
    phi_b = np.zeros(phi.shape)
    phi_f = np.zeros(phi.shape)

    for n in range(3, len(phi) - 3):
        phi_b[3:-3, n] = delta_plus_y(phi, n - 1)

    for n in range(3, len(phi) - 3):
        phi_f[3:-3, n] = delta_minus_y(phi_b, n - 1)

    d = phi_f / dy

    return a, b, c, d


def iso_bx(phi, dx):
    is0 = np.zeros(phi.shape)
    is1 = np.zeros(phi.shape)
    is2 = np.zeros(phi.shape)

    a, b, c, d = parameters_bx(phi, dx)

    is0[3:-3, 3:-3] = 13 * (a[3:-3, 3:-3] - b[3:-3, 3:-3]) ** 2 + 3 * (a[3:-3, 3:-3] - 3 * b[3:-3, 3:-3]) ** 2
    is1[3:-3, 3:-3] = 13 * (b[3:-3, 3:-3] - c[3:-3, 3:-3]) ** 2 + 3 * (b[3:-3, 3:-3] + c[3:-3, 3:-3]) ** 2
    is2[3:-3, 3:-3] = 13 * (c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2 + 3 * (3 * c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2

    return is0, is1, is2


def iso_fx(phi, dx):
    is0 = np.zeros(phi.shape)
    is1 = np.zeros(phi.shape)
    is2 = np.zeros(phi.shape)

    a, b, c, d = parameters_fx(phi, dx)

    is0[3:-3, 3:-3] = 13 * (a[3:-3, 3:-3] - b[3:-3, 3:-3]) ** 2 + 3 * (a[3:-3, 3:-3] - 3 * b[3:-3, 3:-3]) ** 2
    is1[3:-3, 3:-3] = 13 * (b[3:-3, 3:-3] - c[3:-3, 3:-3]) ** 2 + 3 * (b[3:-3, 3:-3] + c[3:-3, 3:-3]) ** 2
    is2[3:-3, 3:-3] = 13 * (c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2 + 3 * (3 * c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2

    return is0, is1, is2


def iso_by(phi, dy):
    is0 = np.zeros(phi.shape)
    is1 = np.zeros(phi.shape)
    is2 = np.zeros(phi.shape)

    a, b, c, d = parameters_by(phi, dy)

    is0[3:-3, 3:-3] = 13 * (a[3:-3, 3:-3] - b[3:-3, 3:-3]) ** 2 + 3 * (a[3:-3, 3:-3] - 3 * b[3:-3, 3:-3]) ** 2
    is1[3:-3, 3:-3] = 13 * (b[3:-3, 3:-3] - c[3:-3, 3:-3]) ** 2 + 3 * (b[3:-3, 3:-3] + c[3:-3, 3:-3]) ** 2
    is2[3:-3, 3:-3] = 13 * (c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2 + 3 * (3 * c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2

    return is0, is1, is2


def iso_fy(phi, dy):
    is0 = np.zeros(phi.shape)
    is1 = np.zeros(phi.shape)
    is2 = np.zeros(phi.shape)

    a, b, c, d = parameters_fy(phi, dy)

    is0[3:-3, 3:-3] = 13 * (a[3:-3, 3:-3] - b[3:-3, 3:-3]) ** 2 + 3 * (a[3:-3, 3:-3] - 3 * b[3:-3, 3:-3]) ** 2
    is1[3:-3, 3:-3] = 13 * (b[3:-3, 3:-3] - c[3:-3, 3:-3]) ** 2 + 3 * (b[3:-3, 3:-3] + c[3:-3, 3:-3]) ** 2
    is2[3:-3, 3:-3] = 13 * (c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2 + 3 * (3 * c[3:-3, 3:-3] - d[3:-3, 3:-3]) ** 2

    return is0, is1, is2


def al_bx(phi, dx):
    e = 10 ** (-6)

    a0 = np.zeros(phi.shape)
    a1 = np.zeros(phi.shape)
    a2 = np.zeros(phi.shape)

    is0, is1, is2 = iso_bx(phi, dx)

    a0[3:-3, 3:-3] = 1 / (e + is0[3:-3, 3:-3]) ** 2
    a1[3:-3, 3:-3] = 6 / (e + is1[3:-3, 3:-3]) ** 2
    a2[3:-3, 3:-3] = 3 / (e + is2[3:-3, 3:-3]) ** 2

    return a0, a1, a2


def al_fx(phi, dx):
    e = 10 ** (-6)

    a0 = np.zeros(phi.shape)
    a1 = np.zeros(phi.shape)
    a2 = np.zeros(phi.shape)

    is0, is1, is2 = iso_fx(phi, dx)

    a0[3:-3, 3:-3] = 1 / (e + is0[3:-3, 3:-3]) ** 2
    a1[3:-3, 3:-3] = 6 / (e + is1[3:-3, 3:-3]) ** 2
    a2[3:-3, 3:-3] = 3 / (e + is2[3:-3, 3:-3]) ** 2

    return a0, a1, a2


def al_by(phi, dy):
    e = 10 ** (-6)

    a0 = np.zeros(phi.shape)
    a1 = np.zeros(phi.shape)
    a2 = np.zeros(phi.shape)

    is0, is1, is2 = iso_by(phi, dy)

    a0[3:-3, 3:-3] = 1 / (e + is0[3:-3, 3:-3]) ** 2
    a1[3:-3, 3:-3] = 6 / (e + is1[3:-3, 3:-3]) ** 2
    a2[3:-3, 3:-3] = 3 / (e + is2[3:-3, 3:-3]) ** 2

    return a0, a1, a2


def al_fy(phi, dy):
    e = 10 ** (-6)

    a0 = np.zeros(phi.shape)
    a1 = np.zeros(phi.shape)
    a2 = np.zeros(phi.shape)

    is0, is1, is2 = iso_fy(phi, dy)

    a0[3:-3, 3:-3] = 1 / (e + is0[3:-3, 3:-3]) ** 2
    a1[3:-3, 3:-3] = 6 / (e + is1[3:-3, 3:-3]) ** 2
    a2[3:-3, 3:-3] = 3 / (e + is2[3:-3, 3:-3]) ** 2

    return a0, a1, a2


def weight_bx(phi, dx):
    w0 = np.zeros(phi.shape)
    w2 = np.zeros(phi.shape)

    a0, a1, a2 = al_bx(phi, dx)

    w0[3:-3, 3:-3] = a0[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])
    w2[3:-3, 3:-3] = a2[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])

    return w0, w2


def weight_fx(phi, dx):
    w0 = np.zeros(phi.shape)
    w2 = np.zeros(phi.shape)

    a0, a1, a2 = al_fx(phi, dx)

    w0[3:-3, 3:-3] = a0[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])
    w2[3:-3, 3:-3] = a2[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])

    return w0, w2


def weight_by(phi, dy):
    w0 = np.zeros(phi.shape)
    w2 = np.zeros(phi.shape)

    a0, a1, a2 = al_by(phi, dy)

    w0[3:-3, 3:-3] = a0[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])
    w2[3:-3, 3:-3] = a2[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])

    return w0, w2


def weight_fy(phi, dy):
    w0 = np.zeros(phi.shape)
    w2 = np.zeros(phi.shape)

    a0, a1, a2 = al_fy(phi, dy)

    w0[3:-3, 3:-3] = a0[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])
    w2[3:-3, 3:-3] = a2[3:-3, 3:-3] / (a0[3:-3, 3:-3] + a1[3:-3, 3:-3] + a2[3:-3, 3:-3])

    return w0, w2


def weno_p(a, b, c, d, w0, w2, phi):
    weno1 = np.zeros(phi.shape)

    weno1[3:-3, 3:-3] = (1 / 3) * w0[3:-3, 3:-3] * (a[3:-3, 3:-3] - 2 * b[3:-3, 3:-3] + c[3:-3, 3:-3]) + (1 / 6) * \
                        (w2[3:-3, 3:-3] - (1 / 2)) * (b[3:-3, 3:-3] - 2 * c[3:-3, 3:-3] + d[3:-3, 3:-3])

    return weno1


def unique1(phi, dx):
    eq = np.zeros(phi.shape)
    u1 = np.zeros(phi.shape)
    u2 = np.zeros(phi.shape)
    u3 = np.zeros(phi.shape)
    u4 = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        u1[n, 3:-3] = (delta_plus_x(phi, n-2)) / dx

    for n in range(3, len(phi)-3):
        u2[n, 3:-3] = (delta_plus_x(phi, n-1)) / dx

    for n in range(3, len(phi)-3):
        u3[n, 3:-3] = (delta_plus_x(phi, n+0)) / dx

    for n in range(3, len(phi)-3):
        u4[n, 3:-3] = (delta_plus_x(phi, n+1)) / dx

    eq[3:-3, 3:-3] = (1 / 12) * (-u1[3:-3, 3:-3] + 7 * u2[3:-3, 3:-3] + 7 * u3[3:-3, 3:-3] - u4[3:-3, 3:-3])

    return eq


def unique2(phi, dy):
    eq = np.zeros(phi.shape)
    u1 = np.zeros(phi.shape)
    u2 = np.zeros(phi.shape)
    u3 = np.zeros(phi.shape)
    u4 = np.zeros(phi.shape)

    for n in range(3, len(phi)-3):
        u1[3:-3, n] = (delta_plus_y(phi, n - 2)) / dy

    for n in range(3, len(phi)-3):
        u2[3:-3, n] = (delta_plus_y(phi, n - 1)) / dy

    for n in range(3, len(phi)-3):
        u3[3:-3, n] = (delta_plus_y(phi, n + 0)) / dy

    for n in range(3, len(phi)-3):
        u4[3:-3, n] = (delta_plus_y(phi, n + 1)) / dy

    eq[3:-3, 3:-3] = (1 / 12) * (-u1[3:-3, 3:-3] + 7 * u2[3:-3, 3:-3] + 7 * u3[3:-3, 3:-3] - u4[3:-3, 3:-3])

    return eq


def weno(phi, dx, dy, nodes):
    """
    This function implements the ideas of the following paper:
    [1] G.-S. Jiang and D. Peng, “Weighted ENO Schemes for Hamilton--Jacobi Equations,” SIAM J. Sci. Comput., vol. 21,
    no. 6, pp. 2126–2143, Jan. 2000.
    :param phi:
    :param dx:
    :param dy:
    :param nodes:
    :return:
    """
    padding = 3

    phi = np.pad(phi, padding, 'reflect', reflect_type='odd')

    dxb = np.zeros((nodes, nodes))
    dxf = np.zeros((nodes, nodes))
    dyb = np.zeros((nodes, nodes))
    dyf = np.zeros((nodes, nodes))

    # Backward for x
    phi1 = unique1(phi, dx)
    a, b, c, d = parameters_bx(phi, dx)
    w0, w2 = weight_bx(phi, dx)
    phi2 = weno_p(a, b, c, d, w0, w2, phi)

    dxb[:, :] = phi1[3:-3, 3:-3] - phi2[3:-3, 3:-3]

    # Forward for x
    phi1 = unique1(phi, dx)
    a, b, c, d = parameters_fx(phi, dx)
    w0, w2 = weight_fx(phi, dx)
    phi2 = weno_p(a, b, c, d, w0, w2, phi)

    dxf[:, :] = phi1[3:-3, 3:-3] + phi2[3:-3, 3:-3]

    # Backward for y
    phi3 = unique2(phi, dy)
    a, b, c, d = parameters_by(phi, dy)
    w0, w2 = weight_by(phi, dy)
    phi2 = weno_p(a, b, c, d, w0, w2, phi)

    dyb[:, :] = phi3[3:-3, 3:-3] - phi2[3:-3, 3:-3]

    # Forward for y
    phi3 = unique2(phi, dy)
    a, b, c, d = parameters_fy(phi, dy)
    w0, w2 = weight_fy(phi, dy)
    phi2 = weno_p(a, b, c, d, w0, w2, phi)

    dyf[:, :] = phi3[3:-3, 3:-3] + phi2[3:-3, 3:-3]

    return dxb, dxf, dyb, dyf
