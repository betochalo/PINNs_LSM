import numpy as np


# import sys
# import json


def circle(x, y, xc, yc, r):
    """
    Create a distance function for a circle
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xc: x coordinate for the center
    :param yc: y coordinate for the center
    :param r: radius
    :return: phi distance function
    """
    phi = (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5) - r

    return phi


def rectangle(x, y, xr, yr, a, b):
    """
    Creates a distance function for a rectangle.
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xr: x coordinate for the center
    :param yr: y coordinate for the center
    :param a: Side length    :param b:
    :return: phi   Distance function
    """

    phi1 = -((x - xr) + a / 2)
    phi2 = (x - xr) - a / 2
    phi3 = -((y - yr) + b / 2)
    phi4 = (y - yr) - b / 2
    phix = np.maximum(phi1, phi2)
    phiy = np.maximum(phi3, phi4)
    phi = np.maximum(phix, phiy)

    return phi


def edge_distance(p, a, b):
    pa, ba = p - a, b - a
    h = np.clip(np.sum(pa * ba, axis=-1) / np.sum(ba * ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba * h[..., np.newaxis], axis=-1)


def triangle_sdf(p, a, b, c):
    d1 = edge_distance(p, a, b)
    d2 = edge_distance(p, b, c)
    d3 = edge_distance(p, c, a)

    # Use cross product to determine if point is inside triangle
    sign_a = np.sign(np.cross(b - a, p - a))
    sign_b = np.sign(np.cross(c - b, p - b))
    sign_c = np.sign(np.cross(a - c, p - c))

    inside = ((sign_a >= 0) & (sign_b >= 0) & (sign_c >= 0)) | ((sign_a <= 0) & (sign_b <= 0) & (sign_c <= 0))

    # The SDF value
    d = np.where(inside, -np.minimum(np.minimum(d1, d2), d3), np.minimum(np.minimum(d1, d2), d3))
    return d


def packman(x, y, a, b, c, xc, yc, r, xc1, yc2, r1):
    phi1 = triangle_sdf(np.stack((x, y), axis=-1), a, b, c)
    phi2 = circle(x, y, xc, yc, r)
    phi3 = circle(x, y, xc1, yc2, r1)
    phi4 = np.maximum(phi2, np.negative(phi1))
    phi = np.maximum(phi4, np.negative(phi3))
    return phi


def zalesak_disk(x, y, xc, yc, r, xr, yr, a, b):
    """
    Create a distance function for the zalesak_disk
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xc: x coordinate for the circle
    :param yc: y coordinate for the circle
    :param r: radius
    :param xr: x coordinate for the rectangle
    :param yr: y coordinate for the rectangle
    :param a: side length
    :param b: side length
    :return: distance function
    """
    phi_circle = circle(x, y, xc, yc, r)
    phi_rectangle = rectangle(x, y, xr, yr, a, b)

    phi = np.maximum(phi_circle, np.negative(phi_rectangle))

    return phi


def nut(x, y, xc, yc, r, xr, yr, a, b, xr1, yr1, c, d):
    phi_circle = circle(x, y, xc, yc, r)
    phi_rectangle1 = rectangle(x, y, xr, yr, a, b)
    phi_rectangle2 = rectangle(x, y, xr1, yr1, c, d)
    phi1 = np.maximum(phi_circle, np.negative(phi_rectangle1))
    phi2 = np.maximum(phi1, np.negative(phi_rectangle2))

    return phi2


def sphere(x, y, z, xc, yc, zc, r):
    """
    Create a distance function for a circle
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xc: x coordinate for the center
    :param yc: y coordinate for the center
    :param r: radius
    :return: phi distance function
    """
    # phi = (((x - xc) ** 2 + (y - yc) ** 2) ** 0.5 + (z - zc) ** 2) - r
    phi = ((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2) ** 0.5 - r

    return phi


def rectangle3D(x, y, z, xr, yr, zr, a, b, c):
    """
    Creates a distance function for a rectangle.
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xr: x coordinate for the center
    :param yr: y coordinate for the center
    :param a: Side length
    :param b: Side length
    :return: phi   Distance function
    """

    phi1 = -((x - xr) + a / 2)
    phi2 = (x - xr) - a / 2
    phi3 = -((y - yr) + b / 2)
    phi4 = (y - yr) - b / 2
    phi5 = -((z - zr) + c / 2)
    phi6 = (z - zr) - c / 2
    phix = np.maximum(phi1, phi2)
    phiy = np.maximum(phi3, phi4)
    phiz = np.maximum(phi5, phi6)
    phi = np.maximum(phix, phiy, phiz)

    return phi


def zalesak_disk3D(x, y, z, xc, yc, zc, r, xr, yr, zr, a, b, c):
    """
    Create a distance function for the zalesak_disk
    :param x: X-values from np.meshgrid()
    :param y: Y-values from np.meshgrid()
    :param xc: x coordinate for the circle
    :param yc: y coordinate for the circle
    :param r: radius
    :param xr: x coordinate for the rectangle
    :param yr: y coordinate for the rectangle
    :param a: side length
    :param b: side length
    :return: distance function
    """
    phi_circle = sphere(x, y, z, xc, yc, zc, r)
    phi_rectangle = rectangle3D(x, y, z, xr, yr, zr, a, b, c)

    phi = np.maximum(phi_circle, np.negative(phi_rectangle))

    return phi
