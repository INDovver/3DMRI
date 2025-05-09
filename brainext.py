import numpy as np
import nibabel as nib
import trimesh
from numba import jit
from numba.typed import List
from aditionals import createSphere, getClosestIntegerPoint, bresenham3D, calculateL2Norm, calculatel2NormArray, calculateDiagonalDot


class BrainExtractor:
    def __init__(
        self,
        img,
        t02t = 0.02,
        t98t = 0.98,
        bt = 0.6,
        d1 = 20.0,
        d2 = 10.0,
        rmin = 3.33,
        rmax = 10.0,
    ):
        res = img.header["pixdim"][1]

        self.bt = bt
        self.d1 = d1 / res
        self.d2 = d2 / res
        self.rmin = rmin / res
        self.rmax = rmax / res

        self.E = (1.0 / self.rmin + 1.0 / self.rmax) / 2.0
        self.F = 6.0 / (1.0 / self.rmin - 1.0 / self.rmax)

        self.img = img

        self.data = img.get_fdata() 
        self.rdata = img.get_fdata().ravel()
        self.shape = img.shape  
        self.rshape = np.multiply.reduce(img.shape)  

        sortedData = np.sort(self.rdata)
        self.tmin = np.min(sortedData)
        self.t2 = sortedData[np.ceil(t02t * self.rshape).astype(np.int64) + 1]
        self.t98 = sortedData[np.ceil(t98t * self.rshape).astype(np.int64) + 1]
        self.tmax = np.max(sortedData)
        self.t = (self.t98 - self.t2) * 0.1 + self.t2

        ic, jc, kc = np.meshgrid(
            np.arange(self.shape[0]), np.arange(self.shape[1]), np.arange(self.shape[2]), indexing="ij", copy=False
        )
        cdata = np.clip(self.rdata, self.t2, self.t98) * (self.rdata > self.t)
        ci = np.average(ic.ravel(), weights=cdata)
        cj = np.average(jc.ravel(), weights=cdata)
        ck = np.average(kc.ravel(), weights=cdata)
        self.c = np.array([ci, cj, ck])

        self.r = 0.5 * np.cbrt(3 * np.sum(self.rdata > self.t) / (4 * np.pi))

        self.tm = np.median(self.data[createSphere(self.shape, 2 * self.r, self.c)])

        self.surface = trimesh.creation.icosphere(subdivisions=4, radius=self.r)
        self.surface = self.surface.apply_transform([[1, 0, 0, ci], [0, 1, 0, cj], [0, 0, 1, ck], [0, 0, 0, 1]])

        self.numVertices = self.surface.vertices.shape[0]
        self.numFaces = self.surface.faces.shape[0]
        self.vertices = np.array(self.surface.vertices)
        self.faces = np.array(self.surface.faces)
        self.vertexNeighborsIdx = List([np.array(i) for i in self.surface.vertex_neighbors])
        self.faceVertexIdxs = np.zeros((self.numVertices, 6, 2), dtype=np.int64)
        for v in range(self.numVertices):
            f, i = np.asarray(self.faces == v).nonzero()
            self.faceVertexIdxs[v, : i.shape[0], 0] = f
            self.faceVertexIdxs[v, : i.shape[0], 1] = i
            if i.shape[0] == 5:
                self.faceVertexIdxs[v, 5, 0] = -1
                self.faceVertexIdxs[v, 5, 1] = -1
        self.updateSurfaceAttributes()

    @staticmethod
    @jit(nopython=True, cache=True)
    def computeFaceNormals(numFaces, faces, vertices):
        faceNormals = np.zeros((numFaces, 3))
        for i, f in enumerate(faces):
            localV = vertices[f]
            a = localV[1] - localV[0]
            b = localV[2] - localV[0]
            faceNormals[i] = np.array(
                (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            )
            faceNormals[i] /= calculateL2Norm(faceNormals[i])
        return faceNormals

    @staticmethod
    def computeFaceAngles(triangles):
        triangles = np.asanyarray(triangles, dtype=np.float64)

        u = triangles[:, 1] - triangles[:, 0]
        u /= calculatel2NormArray(u)[:, np.newaxis]
        v = triangles[:, 2] - triangles[:, 0]
        v /= calculatel2NormArray(v)[:, np.newaxis]
        w = triangles[:, 2] - triangles[:, 1]
        w /= calculatel2NormArray(w)[:, np.newaxis]

        result = np.zeros((len(triangles), 3), dtype=np.float64)
        result[:, 0] = np.arccos(np.clip(calculateDiagonalDot(u, v), -1, 1))
        result[:, 1] = np.arccos(np.clip(calculateDiagonalDot(-u, w), -1, 1))
        result[:, 2] = np.pi - result[:, 0] - result[:, 1]

        result[(result < 1e-8).any(axis=1), :] = 0.0
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def computeVertexNormals(
        numVertices,
        faceNormals,
        faceAngles,
        faceVertexIdxs,
    ):
        vertexNormals = np.zeros((numVertices, 3))
        for vertexIdx in range(numVertices):
            faceIdxs = np.asarray([f for f in faceVertexIdxs[vertexIdx, :, 0] if f != -1])
            infaceIdxs = np.asarray([f for f in faceVertexIdxs[vertexIdx, :, 1] if f != -1])
            surroundingAngles = faceAngles.ravel()[faceIdxs * 3 + infaceIdxs]
            vertexNormals[vertexIdx] = np.dot(surroundingAngles / surroundingAngles.sum(), faceNormals[faceIdxs])
            vertexNormals[vertexIdx] /= calculateL2Norm(vertexNormals[vertexIdx])
        return vertexNormals

    def rebuildSurface(self):
        self.updateSurfaceAttributes()
        self.surface = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    @staticmethod
    @jit(nopython=True, cache=True)
    def updateSurfAttr(vertices, neighborsIdx):
        neighbors = np.zeros((vertices.shape[0], 6, 3))
        neighborsSize = np.zeros(vertices.shape[0], dtype=np.int8)
        for i, ni in enumerate(neighborsIdx):
            for j, vi in enumerate(ni):
                neighbors[i, j, :] = vertices[vi]
            neighborsSize[i] = j + 1

        centroids = np.zeros((vertices.shape[0], 3))
        for i, (n, s) in enumerate(zip(neighbors, neighborsSize)):
            centroids[i, 0] = np.mean(n[:s, 0])
            centroids[i, 1] = np.mean(n[:s, 1])
            centroids[i, 2] = np.mean(n[:s, 2])

        return neighbors, neighborsSize, centroids

    def updateSurfaceAttributes(self):
        self.triangles = self.vertices[self.faces]
        self.faceNormals = self.computeFaceNormals(self.numFaces, self.faces, self.vertices)
        self.faceAngles = self.computeFaceAngles(self.triangles)
        self.vertexNormals = self.computeVertexNormals(
            self.numVertices, self.faceNormals, self.faceAngles, self.faceVertexIdxs
        )
        self.vertexNeighbors, self.vertexNeighborsSize, self.vertexNeighborsCentroids = self.updateSurfAttr(
            self.vertices, self.vertexNeighborsIdx
        )
        self.l = self.getMeanIntervertexDistance(self.vertices, self.vertexNeighbors, self.vertexNeighborsSize)

    @staticmethod
    @jit(nopython=True, cache=True)
    def getMeanIntervertexDistance(vertices, neighbors, sizes):
        mivd = np.zeros(vertices.shape[0])
        for v in range(vertices.shape[0]):
            vecs = vertices[v] - neighbors[v, : sizes[v]]
            vd = np.zeros(vecs.shape[0])
            for i in range(vecs.shape[0]):
                vd[i] = calculateL2Norm(vecs[i])
            mivd[v] = np.mean(vd)
        return np.mean(mivd)

    def run(self, iterations = 1000):
        sVectors = np.zeros(self.vertices.shape)

        sN = np.zeros(self.vertices.shape)
        sT = np.zeros(self.vertices.shape)

        u1 = np.zeros(self.vertices.shape)
        u2 = np.zeros(self.vertices.shape)
        u3 = np.zeros(self.vertices.shape)
        u = np.zeros(self.vertices.shape)

        for _ in range(iterations):
            self.stepOfDeformation(
                self.data,
                self.vertices,
                self.vertexNormals,
                self.vertexNeighborsCentroids,
                self.l,
                self.t2,
                self.t,
                self.tm,
                self.E,
                self.F,
                self.bt,
                self.d1,
                self.d2,
                sVectors,
                sN,
                sT,
                u1,
                u2,
                u3,
                u,
            )
            self.vertices += u
            self.updateSurfaceAttributes()

        self.rebuildSurface()

    @staticmethod
    @jit(nopython=True, cache=True)
    def stepOfDeformation(
        data,
        vertices,
        normals,
        neighborsCentroids,
        l,
        t2,
        t,
        tm,
        E,
        F,
        bt,
        d1,
        d2,
        sVectors,
        sN,
        sT,
        u1,
        u2,
        u3,
        u,
    ):
        for i, vertex in enumerate(vertices):
            sVectors[i] = neighborsCentroids[i] - vertex

            sN[i] = np.dot(sVectors[i], normals[i]) * normals[i]
            sT[i] = sVectors[i] - sN[i]

            u1[i] = 0.5 * sT[i]

            rCurvature = (l ** 2) / (2 * calculateL2Norm(sN[i]))

            f2 = (1 + np.tanh(F * (1 / rCurvature - E))) / 2

            u2[i] = f2 * sN[i]

            e1 = getClosestIntegerPoint(vertex - d1 * normals[i])
            e2 = getClosestIntegerPoint(vertex - d2 * normals[i])

            cPoint = getClosestIntegerPoint(vertex)
            i1Line = bresenham3D(cPoint, e1)
            i2Line = bresenham3D(cPoint, e2)

            linedata1 = []
            for dPoint in i1Line:
                if (
                    0 <= dPoint[0] < data.shape[0]
                    and 0 <= dPoint[1] < data.shape[1]
                    and 0 <= dPoint[2] < data.shape[2]
                ):
                    linedata1.append(data[dPoint[0], dPoint[1], dPoint[2]])
            linedata1.append(tm)
            linedata1 = np.asarray(linedata1)
            Imin = np.max(np.asarray([t2, np.min(linedata1)]))

            linedata2 = []
            for dPoint in i2Line:
                if (
                    0 <= dPoint[0] < data.shape[0]
                    and 0 <= dPoint[1] < data.shape[1]
                    and 0 <= dPoint[2] < data.shape[2]
                ):
                    linedata2.append(data[dPoint[0], dPoint[1], dPoint[2]])
            linedata2.append(t)
            linedata2 = np.asarray(linedata2)
            Imax = np.min(np.asarray([tm, np.max(linedata2)]))

            tl = (Imax - t2) * bt + t2

            f3 = 0.05 * 2 * (Imin - tl) / (Imax - t2) * l

            u3[i] = f3 * normals[i]

        u[:, :] = u1 + u2 + u3

    @staticmethod
    def checkBound(imgMin, imgMax, imgStart, imgEnd, volStart, volEnd):
        if imgMin < imgStart:
            volStart = volStart + (imgStart - imgMin)
            imgMin = 0
        if imgMax > imgEnd:
            volEnd = volEnd - (imgMax - imgEnd)
            imgMax = imgEnd
        return imgMin, imgMax, imgStart, imgEnd, volStart, volEnd

    def computeMask(self):
        vol = self.surface.voxelized(1)
        vol = vol.fill()
        self.mask = np.zeros(self.shape)

        xMin = int(vol.bounds[0, 0]) if vol.bounds[0, 0] > 0 else int(vol.bounds[0, 0]) - 1
        xMax = int(vol.bounds[1, 0]) if vol.bounds[1, 0] > 0 else int(vol.bounds[1, 0]) - 1
        yMin = int(vol.bounds[0, 1]) if vol.bounds[0, 1] > 0 else int(vol.bounds[0, 1]) - 1
        yMax = int(vol.bounds[1, 1]) if vol.bounds[1, 1] > 0 else int(vol.bounds[1, 1]) - 1
        zMin = int(vol.bounds[0, 2]) if vol.bounds[0, 2] > 0 else int(vol.bounds[0, 2]) - 1
        zMax = int(vol.bounds[1, 2]) if vol.bounds[1, 2] > 0 else int(vol.bounds[1, 2]) - 1

        xStart = 0
        yStart = 0
        zStart = 0
        xEnd = int(self.shape[0])
        yEnd = int(self.shape[1])
        zEnd = int(self.shape[2])

        xVolStart = 0
        yVolStart = 0
        zVolStart = 0
        xVolEnd = int(vol.matrix.shape[0])
        yVolEnd = int(vol.matrix.shape[1])
        zVolEnd = int(vol.matrix.shape[2])

        xMin, xMax, xStart, xEnd, xVolStart, xVolEnd = self.checkBound(
            xMin, xMax, xStart, xEnd, xVolStart, xVolEnd
        )
        yMin, yMax, yStart, yEnd, yVolStart, yVolEnd = self.checkBound(
            yMin, yMax, yStart, yEnd, yVolStart, yVolEnd
        )
        zMin, zMax, zStart, zEnd, zVolStart, zVolEnd = self.checkBound(
            zMin, zMax, zStart, zEnd, zVolStart, zVolEnd
        )
        self.mask[xMin:xMax, yMin:yMax, zMin:zMax] = vol.matrix[
            xVolStart:xVolEnd, yVolStart:yVolEnd, zVolStart:zVolEnd
        ]
        return self.mask

    def saveMask(self, filename):
        maskResult = self.computeMask()
        nib.Nifti1Image(maskResult, self.img.affine).to_filename(filename)