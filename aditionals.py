import numpy as np
from numba import jit


def createSphere(shape, radius, position):
    semisizes = (radius,) * 3

    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    positionGrid = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for xI, semisize in zip(positionGrid, semisizes):
        arr += (xI / semisize) ** 2

    return arr <= 1.0

@jit(nopython=True, cache=True)
def getClosestIntegerPoint(vertex):
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    x0 = np.floor(x)
    y0 = np.floor(y)
    z0 = np.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    minEuclid = 99

    for i in [x0, x1]:
        for j in [y0, y1]:
            for k in [z0, z1]:
                coords = np.array([i, j, k])
                dist = calculateL2Norm(vertex - coords)
                if dist < minEuclid:
                    minEuclid = dist
                    finalCoords = coords

    return finalCoords.astype(np.int64)

@jit(nopython=True, cache=True)
def bresenham3D(v0, v1):
    dx = np.abs(v1[0] - v0[0])
    dy = np.abs(v1[1] - v0[1])
    dz = np.abs(v1[2] - v0[2])
    xs = 1 if (v1[0] > v0[0]) else -1
    ys = 1 if (v1[1] > v0[1]) else -1
    zs = 1 if (v1[2] > v0[2]) else -1

    if dx >= dy and dx >= dz:
        d0 = dx
        d1 = dy
        d2 = dz
        s0 = xs
        s1 = ys
        s2 = zs
        a0 = 0
        a1 = 1
        a2 = 2
    elif dy >= dx and dy >= dz:
        d0 = dy
        d1 = dx
        d2 = dz
        s0 = ys
        s1 = xs
        s2 = zs
        a0 = 1
        a1 = 0
        a2 = 2
    elif dz >= dx and dz >= dy:
        d0 = dz
        d1 = dx
        d2 = dy
        s0 = zs
        s1 = xs
        s2 = ys
        a0 = 2
        a1 = 0
        a2 = 1

    line = np.zeros((d0 + 1, 3), dtype=np.int64)
    line[0] = v0

    p1 = 2 * d1 - d0
    p2 = 2 * d2 - d0
    for i in range(d0):
        c = line[i].copy()
        c[a0] += s0
        if p1 >= 0:
            c[a1] += s1
            p1 -= 2 * d0
        if p2 >= 0:
            c[a2] += s2
            p2 -= 2 * d0
        p1 += 2 * d1
        p2 += 2 * d2
        line[i + 1] = c

    return line

@jit(nopython=True, cache=True)
def calculateL2Norm(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

@jit(nopython=True, cache=True)
def calculatel2NormArray(array):
    return np.sqrt(array[:, 0] ** 2 + array[:, 1] ** 2 + array[:, 2] ** 2)

def calculateDiagonalDot(a, b):
    a = np.asanyarray(a)
    return np.dot(a * b, [1.0] * a.shape[1])