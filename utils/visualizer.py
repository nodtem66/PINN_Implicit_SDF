# -*- coding: utf-8 -*-

from sdf.mesh import _cartesian_product, _estimate_bounds
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
import torch
import time

class SDFVisualize():
    def __init__(self, z_level=0, step=0.01, offset=30, nums=100, tag=None, writer=None):
        self.z_level = z_level
        self.offset = offset
        self.nums = nums
        self.step =step
        self.tag = tag if tag is not None else str(int(time.time()))
        self.writer = writer

    def from_nn(self, model, bounds=[(-1,-1,-1), (1,1,1)], bounds_from_mesh=None, device='cpu', plot_gradient=False):
        dx, dy, dz = (self.step,)*3
        (x0, y0, z0), (x1, y1, z1) = self._bounds_from_mesh(bounds_from_mesh) if not bounds_from_mesh is None else bounds 


        # X = np.arange(x0-offset*dx, x1+offset*dx, dx)
        # Y = np.arange(y0-offset*dy, y1+offset*dy, dy)
        # Z = np.arange(z0-offset*dz, z1+offset*dz, dz)

        X = np.linspace(x0-self.offset*dx, x1+self.offset*dx, self.nums)
        Y = np.linspace(y0-self.offset*dy, y1+self.offset*dy, self.nums)
        Z = np.array([self.z_level])

        P = _cartesian_product(X, Y, Z)
        sdf = model.forward(torch.from_numpy(P).to(device)).cpu().detach().numpy().reshape((self.nums, self.nums))
        if plot_gradient:
            gradient = model.get_gradient(torch.from_numpy(P).to(device)).cpu().detach().numpy().reshape((self.nums, self.nums))
            gradient_2 = model.get_gradient2(torch.from_numpy(P).to(device)).cpu().detach().numpy()
            gradient_2 = np.linalg.norm(gradient_2, axis=1).reshape((self.nums, self.nums))

        if self.writer is not None:
            plt.switch_backend('agg')
        fig1, ax = plt.subplots()
        plt.pcolormesh(sdf, cmap='coolwarm', norm=CenteredNorm())
        plt.colorbar()
        ax.set_title('Predict SDF\n(min=%.6f max=%.6f)' % (np.min(sdf), np.max(sdf)))
        ax.tick_params(axis='both', which='major', labelsize=6)
        if self.writer is None:
            plt.show()
        
        if plot_gradient:
            fig2, ax = plt.subplots()
            plt.pcolormesh(gradient - 1.0, cmap='coolwarm', norm=CenteredNorm())
            plt.colorbar()
            ax.set_title('Predict |$\\nabla(SDF)$| - 1\n(min=%.6f max=%.6f)' % (np.min(gradient - 1.0), np.max(gradient - 1.0)))
            ax.tick_params(axis='both', which='major', labelsize=6)
            if self.writer is None:
                plt.show()
  
            fig3, ax = plt.subplots()
            plt.pcolormesh(gradient_2, cmap='coolwarm', norm=CenteredNorm())
            plt.colorbar()
            ax.set_title('Predict $|\\nabla{|\\nabla(SDF)|}|$\n(min=%.6f max=%.6f)' % (np.min(gradient_2), np.max(gradient_2)))
            ax.tick_params(axis='both', which='major', labelsize=6)
            if self.writer is None:
                plt.show()

        if (self.writer is not None):
            self.writer.add_figure(self.tag+'/predict_sdf', [fig1, fig2, fig3], close=True)


    def from_mesh(self, file, sign_type=4, plot_gradient=False):

        from utils.libs import igl

        v,f = igl.read_triangle_mesh(file)
        bv, bf = igl.bounding_box(v)

        dx, dy, dz = (self.step,)*3
        (x0, y0, z0), (x1, y1, z1) = bv[0], bv[-1]

        X = np.linspace(x0-self.offset*dx, x1+self.offset*dx, self.nums)
        Y = np.linspace(y0-self.offset*dy, y1+self.offset*dy, self.nums)
        Z = np.array([self.z_level])

        P = _cartesian_product(X, Y, Z)
        
        sdf, _, _ = igl.signed_distance(P, v, f, sign_type, return_normals=False)
        sdf = sdf.reshape((self.nums, self.nums))
        
        if plot_gradient:
            gradient = np.linalg.norm(
                np.array(
                    np.gradient(sdf, X[1]-X[0], Y[1]-Y[0])
                ),
                axis=0
            )
            
            gradient_2 = np.linalg.norm(
                np.array(
                    np.gradient(gradient, X[1]-X[0], Y[1]-Y[0])
                ),
                axis=0
            )
           

        if self.writer is not None:
            plt.switch_backend('agg')

        fig1, ax = plt.subplots()
        plt.pcolormesh(sdf, cmap='coolwarm', norm=CenteredNorm())
        plt.colorbar()
        ax.set_title('True SDF\n(min=%.6f max=%.6f)' % (np.min(sdf), np.max(sdf)))
        ax.tick_params(axis='both', which='major', labelsize=6)
        if self.writer is None:
            plt.show()
        
        if plot_gradient:
            fig2, ax = plt.subplots()
            plt.pcolormesh(gradient - 1.0, cmap='coolwarm', norm=CenteredNorm())
            plt.colorbar()
            ax.set_title('True |$\\nabla(SDF)$| - 1\n(min=%.6f max=%.6f)' % (np.min(gradient - 1.0), np.max(gradient - 1.0)))
            ax.tick_params(axis='both', which='major', labelsize=6)
            if self.writer is None:
                plt.show()
  
            fig3, ax = plt.subplots()
            plt.pcolormesh(gradient_2, cmap='coolwarm', norm=CenteredNorm())
            plt.colorbar()
            ax.set_title('True $|\\nabla{|\\nabla(SDF)|}|$\n(min=%.6f max=%.6f)' % (np.min(gradient_2), np.max(gradient_2)))
            ax.tick_params(axis='both', which='major', labelsize=6)
            if self.writer is None:
                plt.show()
        
        if (self.writer is not None):
            self.writer.add_figure(self.tag+'/true_sdf', [fig1, fig2, fig3], close=True)

    def _bounds_from_mesh(self, file):
        from utils.libs import igl
        
        v, _ = igl.read_triangle_mesh(file)
        bv, _ = igl.bounding_box(v)
        return (bv[0], bv[-1])

def plot_model_weight(model, writer=None):
    for key, val in model.model.state_dict().items():
        #print(key, val.max(), val.min(), val.shape)
        d = val.cpu().numpy()
        #print(key, np.min(d), np.max(d))
        if len(d.shape) == 1:
            d = d.reshape(-1, 1)
        fig, ax = plt.subplots()
        plt.imshow(d, cmap='coolwarm')
        ax.set_title('%s\nmin=%.6f\nmax=%.6f' % (key, np.min(d), np.max(d)))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks(np.arange(d.shape[1]))
        ax.set_yticks(np.arange(d.shape[0]))
        ax.set_xticklabels([x for x in range(d.shape[1])])
        ax.set_yticklabels([x for x in range(d.shape[0])])
        fig.tight_layout()
        plt.colorbar()
        plt.show()