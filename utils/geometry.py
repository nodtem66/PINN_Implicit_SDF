# -*- coding: utf-8 -*-
# Original code From On Effectiveness of 
# https://github.com/u2ni/ICML2021/blob/main/neuralImplicitTools/src/geometry.py
# author: 
# 
# Modified by Jirawat to match the current-version pyigl (2.2) that use numpy instead of eigen
import math
from decimal import *

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

from .pyigl_import import igl


class PointSampler(): 
    def __init__(self, mesh, ratio = 0.0, std=0.0, verticeSampling=False, importanceSampling=False):
        self._V = mesh.V()
        self._F = mesh.F()
        self._sampleVertices = verticeSampling

        if ratio < 0 or ratio > 1:
            raise(ValueError("Ratio must be [0,1]"))
        
        self._ratio = ratio
        
        if std < 0 or std > 1:
            raise(ValueError("Normal deviation must be [0,1]"))

        self._std = std

        self._calculateFaceBins()

    def _calculateFaceBins(self):
        """Calculates and saves face area bins for sampling against"""
        vc = np.cross(
            self._V[self._F[:, 0], :] - self._V[self._F[:, 2], :],
            self._V[self._F[:, 1], :] - self._V[self._F[:, 2], :])

        A = np.sqrt(np.sum(vc ** 2, 1))
        FA = A / np.sum(A)
        self._faceBins = np.concatenate(([0],np.cumsum(FA))) 

    def _surfaceSamples(self,n):
        """Returns n points uniformly sampled from surface of mesh"""
        R = np.random.rand(n)   #generate number between [0,1]
        sampleFaceIdxs = np.array(np.digitize(R,self._faceBins)) -1

        #barycentric coordinates for each face for each sample :)
        #random point within face for each sample
        r = np.random.rand(n, 2)
        A = self._V[self._F[sampleFaceIdxs, 0], :]
        B = self._V[self._F[sampleFaceIdxs, 1], :]
        C = self._V[self._F[sampleFaceIdxs, 2], :]
        P = (1 - np.sqrt(r[:,0:1])) * A \
                + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B \
                + np.sqrt(r[:,0:1]) * r[:,1:] * C

        return P

    def _verticeSamples(self, n):
        """Returns n random vertices of mesh"""
        verts = np.random.choice(len(self._V), n)
        return self._V[verts]

    def _normalDist(self, V):
        """Returns normal distribution about each point V"""
        if self._std > 0.0:
            return np.random.normal(loc = V,scale = self._std)

        return V
        
    def _randomSamples(self, n):
        """Returns n random points in unit sphere"""
        # we want to return points in unit sphere, could do using spherical coords
        #   but rejection method is easier and arguably faster :)
        points = np.array([])
        while points.shape[0] < n:
            remainingPoints = n - points.shape[0]
            p = (np.random.rand(remainingPoints,3) - 0.5)*2
            #p = p[np.linalg.norm(p, axis=1) <= SAMPLE_SPHERE_RADIUS]

            if points.size == 0:
                points = p 
            else:
                points = np.concatenate((points, p))
        return points

    def sample(self,n):
        """Returns n points according to point sampler settings"""

        nRandom = round(Decimal(n)*Decimal(self._ratio))
        nSurface = n - nRandom

        xRandom = self._randomSamples(nRandom)

        if nSurface > 0:
            if self._sampleVertices:
                # for comparison later :)
                xSurface = self._verticeSamples(nSurface)
            else:
                xSurface = self._surfaceSamples(nSurface)

            xSurface = self._normalDist(xSurface)
            if nRandom > 0:
                x = np.concatenate((xSurface,xRandom))
            else:
                x = xSurface
        else:
            x = xRandom

        np.random.shuffle(x)    #remove bias on order

        return x

class ImportanceSampler():
    # M, initital uniform set size, N subset size.
    def __init__(self, mesh, M, W=10):
        self.M = M # uniform sample set size
        self.W = W # sample weight...

        if (not mesh is None):
            #if mesh given, we can create our own uniform sampler
            self.uniformSampler = PointSampler(mesh, ratio=1.0) # uniform sampling
            self.sdf = SDF(mesh)
        else:
            # otherwise we assume uniform samples (and the sdf val) will be passed in.
            self.uniformSampler = None 
            self.sdf = None

    def _subsample(self, s, N):

        # weighted by exp distance to surface
        w = np.exp(-self.W*np.abs(s))
        # probabilities to choose each
        pU = w / (np.sum(w) + 1e-9)
        # exclusive sum
        C = np.concatenate(([0],np.cumsum(pU)))
        C = C[0:-1]

        # choose N random buckets
        R = np.random.rand(N)

        # histc
        I = np.array(np.digitize(R,C)) - 1

        return I


    ''' importance sample a given mesh, M uniform samples, N subset based on importance'''
    def sample(self, N):
        if (self.uniformSampler is None):
            raise("No mesh supplied, cannot run importance sampling...")
        if N >= self.M:
            self.M = N*10
        #uniform samples
        U = self.uniformSampler.sample(self.M)
        s = self.sdf.query(U)
        I = self._subsample(s, N)

        #R = np.random.choice(len(U), int(N*0.1))
        S = U[I,:] #np.concatenate((U[I,:],U[R, :]), axis=0)
        return S

    ''' sampling against a supplied U set, where s is sdf at each U'''
    def sampleU(self, N, U, s):
        I = self._subsample(s, N)
        return U[I,:], s[I]

class ImportanceImplicitSampler():
    # M, initital uniform set size, N subset size.
    def __init__(self, sdfs, beta=10):
        if len(sdfs) <= 0:
            raise ValueError('sdfs must be array with one item at least')
        self.beta = beta # sample weight
        # weighted by exp distance to surface
        w = np.exp(-self.beta*np.abs(sdfs))
        # probabilities to choose each
        norm_w = w / (np.sum(w) + 1e-9)
        # exclusive sum
        self.bins = np.concatenate(([0],np.cumsum(norm_w)))
        self.bins = self.bins[0:-1]
 

    ''' importance sample a given mesh, M uniform samples, N subset based on importance'''
    def sample(self, N):
        R = np.random.rand(N)
        I = np.array(np.digitize(R, self.bins)) - 1
        return I

class Mesh():
    _V = np.array([])
    _F = np.array([])
    _normalized = False

    def __init__(
        self, 
        meshPath=None, 
        V=None, 
        F=None,
        viewer = None, 
        doNormalize = True):

        if meshPath is None:
            if V is None or F is None:
                raise("Mesh path or Mesh data must be given")
            else:
                self._V = V
                self._F = F
        else:
            self._loadMesh(meshPath,doNormalize)

        self._viewer = viewer

    def _loadMesh(self, fp, doNormalize):
        #load mesh
        self._V, self._F = igl.read_triangle_mesh(fp)

        if doNormalize:
            self._normalizeMesh()
        
    def V(self):
        return self._V.copy()

    def F(self):
        return self._F.copy()

    def _normalizeMesh(self):
        bv, _ = igl.bounding_box(self._V)
        center = np.sum(bv, axis=0)
        diagonal =igl.bounding_box_diagonal(self._V)
        #scale = 1.2 / diagonal * 2
        
        self._V -= center

    def bounding_box():
        pass

    def show(self, doLaunch = True):
        pass

    def save(self, fp='out.obj'):
        igl.write_triangle_mesh(fp, self._V, self._F, force_ascii=False)

class SDF():
    # Enum definitions
    SIGNED_DISTANCE_TYPE_PSEUDONORMAL = 0           # Use fast pseudo-normal test [Bærentzen & Aanæs 2005]
    SIGNED_DISTANCE_TYPE_WINDING_NUMBER = 1         # Use winding number [Jacobson, Kavan Sorking-Hornug 2013]
    SIGNED_DISTANCE_TYPE_DEFAULT = 2
    SIGNED_DISTANCE_TYPE_UNSIGNED = 3
    SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER = 4   # Use Fast winding number [Barill, Dickson, Schmidt, Levin, Jacobson 2018]

    def __init__(self, mesh: Mesh, signType = SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER):
        assert(signType >= 0 and signType <= 4)
        self._V = mesh.V()
        self._F = mesh.F()
        self._signType = signType

    def query(self, queries):
        """Returns numpy array of SDF values for each point in queries"""
        return (igl.signed_distance(queries, self._V, self._F, self._signType, False)[0])


def normSDF(S, minVal=None,maxVal=None):
    if minVal is None:
        minVal = np.min(S)
        maxVal = np.max(S)
    
    # we don't shift. Keep 0 at 0.
    #S = np.array([item for sublist in S for item in sublist])

    #S[S<0] = -(S[S<0] / minVal)
    #S[S>0] = (S[S>0] / maxVal)
    #S = (S + 1)/2

    S[S<0] = -0.8
    S[S>0] = 0.8
    
    return S

def createAx(idx):
    subplot = pyplot.subplot(idx, projection='3d')
    subplot.set_xlim((-1,1))
    subplot.set_ylim((-1,1))
    subplot.set_zlim((-1,1))
    subplot.view_init(elev=10, azim=100)
    subplot.axis('off')
    subplot.dist = 8
    return subplot

def createAx2d(idx):
    subplot = pyplot.subplot(idx)
    subplot.set_xlim((-1,1))
    subplot.set_ylim((-1,1))
    subplot.axis('off')
    return subplot

def plotCube(ax):
    # draw cube
    r = [-1, 1]

    from itertools import combinations, product
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="black")

def density(U):
    c = gaussian_kde(np.transpose(U))(np.transpose(U))
    return c

def plotMesh(ax, mesh, N=10000):
    surfaceSampler = PointSampler(mesh, ratio=0.0, std=0.0)
    surfaceSamples = surfaceSampler.sample(N)
    x,y,z = np.hsplit(surfaceSamples,3)
    ax.scatter(x,z,y, c='black', marker='.')

def plotSamples(ax, U, c, vmin = -1, is2d = False):
    x,y,z = np.hsplit(U,3)
    ax.scatter(x,y,z,c=c, marker='.',cmap='coolwarm', norm=None, vmin=vmin, vmax=1)

def importanceSamplingComparisonPlot(mesh,sdf):
    fig = pyplot.figure(figsize=(30,10))
    axUniform = createAx(131)
    axSurface = createAx(132)
    axImportance = createAx(133)

    plotCube(axUniform)
    plotCube(axSurface)
    plotCube(axImportance)

    
    #plotMesh(axUniform,mesh)
    #plotMesh(axImportance,mesh)
    #plotMesh(axSurface,mesh)
    
    # plot uniform sampled 
    uniformSampler = PointSampler(mesh, ratio = 1.0)    
    U = uniformSampler.sample(10000)
    SU = sdf.query(U)
    c = normSDF(SU)
    plotSamples(axUniform, U,c)

    # plot surface + noise sampling
    sampler = PointSampler(mesh, ratio = 0.1, std = 0.01, verticeSampling=False)
    p = sampler.sample(10000)
    S = sdf.query(p)
    c = normSDF(S, np.min(SU), np.max(SU))
    plotSamples(axSurface, p,c)

    # plot importance
    importanceSampler = ImportanceSampler(mesh, 100000, 20)
    p = importanceSampler.sample(10000)
    S = sdf.query(p)
    c = normSDF(S, np.min(SU), np.max(SU))

    plotSamples(axImportance, p,c)

    fig.patch.set_visible(False)

    pyplot.axis('off')
    pyplot.show()

def beforeAndAfterPlot(mesh,sdf):
    fig = pyplot.figure(figsize=(10,10))
    fig.patch.set_visible(False)
    axBefore = createAx(111)
    
    pyplot.axis('off')
    #plotCube(axBefore)
    #plotCube(axAfter)

    # plot importance
    importanceSampler = ImportanceSampler(mesh, 100000, 20)
    p = importanceSampler.sample(10000)
    plotSamples(axBefore, p,'grey')
    pyplot.savefig('before.png', dpi=300, transparent=True)
    
    fig = pyplot.figure(figsize=(10,10))
    fig.patch.set_visible(False)
    axAfter = createAx(111)
    S = sdf.query(p)
    c = normSDF(S)
    plotSamples(axAfter, p,c)
    pyplot.savefig('after.png', dpi=300, transparent=True)


def importanceMotivationPlot(mesh,sdf):

    fig = pyplot.figure(figsize=(10,10))
    axSurface = createAx(111)

    #surface sampling
    sampler = PointSampler(mesh, ratio = 0.0, std = 0.01, verticeSampling=False)
    p = sampler.sample(10000)
    c = density(p)
    maxDensity = np.max(c)
    c = c/maxDensity
    plotSamples(axSurface, p,c, vmin=0)
    #pyplot.show()
    pyplot.savefig('surface.png', dpi=300, transparent=True)


    #vertex sampling
    fig = pyplot.figure(figsize=(10,10))
    axVertex = createAx(111)
    sampler = PointSampler(mesh, ratio = 0.0, std = 0.1, verticeSampling=True)
    p = sampler.sample(10000)
    c = density(p)
    maxDensity = np.max(c)
    c = c/maxDensity
    plotSamples(axVertex, p,c, vmin=0)
    #pyplot.show()
    pyplot.savefig('vertex.png', dpi=300, transparent=True)

    fig = pyplot.figure(figsize=(10,10))
    axImportance = createAx(111)
    
    # importance sampling
    importanceSampler = ImportanceSampler(mesh, 1000000, 50)
    p = importanceSampler.sample(10000)
    c = density(p)
    maxDensity = np.max(c)
    c = c/maxDensity
    plotSamples(axImportance, p, c, vmin = 0)
    #pyplot.show()
    pyplot.savefig('importance.png', dpi=300, transparent=True)



def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Train model to predict sdf of a given mesh, by default visualizes reconstructed mesh to you, and plots loss.')
    parser.add_argument('input_mesh', help='path to input mesh')

    args = parser.parse_args()

    mesh = Mesh(meshPath = args.input_mesh, doNormalize=True)

    # first test mesh is loaded correctly
    #mesh.show()

    # test sdf sampling mesh
    sdf = SDF(mesh)
    
    #cubeMarcher = CubeMarcher()
    #grid = cubeMarcher.createGrid(64)

    #S = sdf.query(grid)
    
    #cubeMarcher.march(grid, S)
    #marchedMesh = cubeMarcher.getMesh()

    #marchedMesh.save()
    #marchedMesh.show()

    importanceSamplingComparisonPlot(mesh, sdf)
    beforeAndAfterPlot(mesh,sdf)
    importanceMotivationPlot(mesh,sdf)

if __name__ == '__main__':
    main()







