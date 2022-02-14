# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from matplotlib.colors import CenteredNorm
from sdf.d3 import SDF3
from sdf.mesh import _cartesian_product, _estimate_bounds
from utils.geometry import get_bounding_box_and_offset
from .dataset_generator import SliceDataset
from .operator import gradient


class SDFVisualize:
    def __init__(self, z_level=0, scale_offset=1.0, nums=100, tag=None, writer=None, export=None, device="cpu"):
        self.z_level = z_level
        self.scale_offset = scale_offset
        self.nums = nums
        self.tag = tag if tag is not None else str(int(time.time()))
        self.writer = writer
        self.device = device
        self.export = export

    def from_nn(self, model, bounds=[(-1, -1, -1), (1, 1, 1)], bounds_from_mesh=None, title="predict"):
        
        (x0, y0, z0), (x1, y1, z1) = (
            self._bounds_from_mesh(bounds_from_mesh) if not bounds_from_mesh is None else bounds
        )
        offset_x = abs(self.scale_offset * (x0 - x1))
        offset_y = abs(self.scale_offset * (y0 - y1))
        offset_z = abs(self.scale_offset * (z0 - z1))
        assert(x0 <= x1 and y0 <= y1 and z0 <= z1 and offset_x >= 0 and offset_y >= 0 and offset_z >= 0)

        X = np.linspace(x0 - offset_x, x1 + offset_x, self.nums)
        Y = np.linspace(y0 - offset_y, y1 + offset_y, self.nums)
        Z = np.array([self.z_level])

        P = _cartesian_product(X, Y, Z).astype(np.float32)
        P = torch.from_numpy(P).to(self.device)
        P.requires_grad = True
        sdf = model(P)

        _gradient = torch.linalg.norm(gradient(sdf, P), dim=1)
        
        if self.writer is not None:
            _gradient_2 = torch.linalg.norm(gradient(_gradient, P), aim=1).cpu().detach().numpy().reshape((self.nums, self.nums))
        _gradient = _gradient.cpu().detach().numpy().reshape((self.nums, self.nums))
        sdf = sdf.cpu().detach().numpy().reshape((self.nums, self.nums))

        if self.writer is not None:
            self.writer.add_figure(self.tag + "/predict_sdf", self._plot(sdf, _gradient, _gradient_2), close=True)
        else:
            self._plot(sdf, _gradient, title=title)

    def from_mesh(self, file, sign_type=4, title="true"):

        from .external_import import igl

        v, f = igl.read_triangle_mesh(file)
        (x0, y0, z0), (x1, y1, z1), (offset_x, offset_y, offset_z) = get_bounding_box_and_offset(v, self.scale_offset)

        X = np.linspace(x0 - offset_x, x1 + offset_x, self.nums)
        Y = np.linspace(y0 - offset_y, y1 + offset_y, self.nums)
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        dz = abs(z1 - z0) / (self.nums - 1)
        Z = np.array([self.z_level-dz, self.z_level, self.z_level+dz])

        P = _cartesian_product(X, Y, Z)

        sdf, _, _ = igl.signed_distance(P, v, f, sign_type, return_normals=False)
        sdf = sdf.reshape((self.nums, self.nums, 3))

        gradient = np.linalg.norm(np.array(np.gradient(sdf, dx, dy, dz)), axis=0)

        if self.writer is not None:
            self.writer.add_figure(self.tag + "/true_sdf", self._plot(sdf[:,:,0], gradient[:,:,0]), close=True)
        else:
            self._plot(sdf[:,:,1], gradient[:,:,1], title=title)

    def compare_model_mesh(self, model:torch.nn.Module, mesh_file:str="", title:str="compare"):
        from .external_import import igl

        v, f = igl.read_triangle_mesh(mesh_file)
        pass


    def from_dataset(self, model, filename):
        slice_dataset = SliceDataset.from_file(filename)
        N = round(slice_dataset.sdfs.shape[0] ** (1 / 2))
        _x = torch.from_numpy(slice_dataset.points).to(self.device)
        _x.requires_grad=True
        sdf = model.forward(_x)
        _gradient = torch.linalg.norm(gradient(sdf, _x), dim=1)
        _gradient_2 = torch.linalg.norm(gradient(_gradient, _x), dim=1).cpu().detach().numpy().reshape((N, N))
        _gradient = _gradient.cpu().detach().numpy().reshape((N,N))
        sdf = sdf.cpu().detach().numpy().reshape((N,N))

        dy = dx = np.linalg.norm(slice_dataset.points[1] - slice_dataset.points[0])
        true_gradient = slice_dataset.gradients.reshape((N, N))
        true_gradient_2 = np.linalg.norm(np.array(np.gradient(true_gradient, dx, dy)), axis=0)

        if self.writer is not None:
            self.writer.add_figure(self.tag + "/predict_sdf", self._plot(sdf, _gradient, _gradient_2), close=True)
            self.writer.add_figure(
                self.tag + "/true_sdf", self._plot(slice_dataset.sdfs, true_gradient, true_gradient_2), close=True
            )
        else:
            self._plot(sdf, _gradient, title='predict')
            self._plot(slice_dataset.sdfs.reshape((N, N)), true_gradient, title='true')

    def from_implicit(self, f: SDF3, bounds=((-1, -1, -1), (1, 1, 1)), bounds_from_mesh=None, title:str="Implicit", fmm:bool=False) -> None:
        
        (x0, y0, z0), (x1, y1, z1) = (
            self._bounds_from_mesh(bounds_from_mesh) if not bounds_from_mesh is None else bounds
        )
        offset_x = abs(self.scale_offset * (x0 - x1))
        offset_y = abs(self.scale_offset * (y0 - y1))
        offset_z = abs(self.scale_offset * (z0 - z1))
        assert(x0 <= x1 and y0 <= y1 and z0 <= z1 and offset_x >= 0 and offset_y >= 0 and offset_z >= 0)

        X = np.linspace(x0- offset_x, x1+ offset_x, self.nums)
        Y = np.linspace(y0- offset_y, y1+ offset_y, self.nums)
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        dz = abs(z1 - z0) / (self.nums - 1)
        Z = np.array([self.z_level-dz, self.z_level, self.z_level+dz])

        P = _cartesian_product(X, Y, Z)
        sdf = f(P)
        sdf = sdf.reshape((self.nums, self.nums, 3))

        if fmm:
            import skfmm
            sdf = skfmm.distance(sdf, dx=(dx,dy,dz))

        gradient = np.linalg.norm(np.array(np.gradient(sdf, dx, dy, dz)), axis=0)

        self._plot(sdf[:,:,1], gradient[:,:,1], title=title)

    def _plot(self, sdf, grad=None, grad2=None, title=""):
        figs = []
        timestamp = int(time.time())

        if self.writer is not None:
            plt.switch_backend("agg")

        fig1, ax = plt.subplots()
        plt.pcolormesh(sdf, cmap="coolwarm", norm=CenteredNorm())
        #plt.pcolormesh(sdf, cmap="coolwarm", vmin=-0.1, vmax=0.7)
        plt.colorbar()
        ax.set_title(title + " SDF\n(min=%.6f max=%.6f)" % (np.min(sdf), np.max(sdf)))
        ax.tick_params(axis="both", which="major", labelsize=6)
        
        if self.writer is None:
            plt.show()
        else:
            figs.append(fig1)
        
        if self.export is not None:
            fig1.savefig(f'{title}_sdf_{timestamp}.{self.export}', dpi=300)

        if grad is not None:
            fig2, ax = plt.subplots()
            plt.pcolormesh(grad - 1.0, cmap="coolwarm", norm=CenteredNorm())
            plt.colorbar()
            ax.set_title(
                title + " |$\\nabla(SDF)$| - 1\n(min=%.6f max=%.6f)" % (np.min(grad - 1.0), np.max(grad - 1.0))
            )
            ax.tick_params(axis="both", which="major", labelsize=6)
            
            if self.writer is None:
                plt.show()
            else:
                figs.append(fig2)
            
            if self.export is not None:
                fig2.savefig(f'{title}_g_{timestamp}.{self.export}', dpi=300)

        if grad2 is not None:
            fig3, ax = plt.subplots()
            plt.pcolormesh(grad2, cmap="coolwarm", norm=CenteredNorm())
            plt.colorbar()
            ax.set_title(
                title + " $|\\nabla{|\\nabla(SDF)|}|$\n(min=%.6f max=%.6f)" % (np.min(grad2), np.max(grad2))
            )
            ax.tick_params(axis="both", which="major", labelsize=6)
            
            if self.writer is None:
                plt.show()
            else:
                figs.append(fig3)
            
            if self.export is not None:
                fig3.savefig(f'{title}_gg_{timestamp}.{self.export}', dpi=300)

        return figs

    def _bounds_from_mesh(self, file):
        from .external_import import igl

        v, _ = igl.read_triangle_mesh(file)
        b0, b1, offset = get_bounding_box_and_offset(v)
        return b0, b1

def plot_residual(model, bounds=[(-1, -1, -1), (1, 1, 1)], title="residual", device='cpu', z_level=0.0, nx=100, ny=100):

    (x0, y0, z0), (x1, y1, z1) = bounds

    X = np.linspace(x0, x1, nx)
    Y = np.linspace(y0, y1, ny)
    Z = np.array([z_level])

    P = _cartesian_product(X, Y, Z).astype(np.float32)
    P = torch.from_numpy(P).to(device)

    P.requires_grad_(True)
    y = model.forward(P)
    p = gradient(y, P)
    norm_p = torch.linalg.norm(p, dim=1)
    residual = (norm_p - 1.0)**2
    residual = residual.cpu().detach().numpy().reshape(nx, ny)
    

    fig1, ax = plt.subplots()
    plt.pcolormesh(residual, cmap="coolwarm")
    plt.colorbar()
    ax.set_title(title + " SDF\n(min=%.6f max=%.6f)" % (np.min(residual), np.max(residual)))
    ax.tick_params(axis="both", which="major", labelsize=6)
    plt.show()

def plot_pdf(model, bounds=[(-1, -1, -1), (1, 1, 1)], title="pdf", device='cpu', z_level=0.0, nx=100, ny=100, epsilon=1e-4):

    (x0, y0, z0), (x1, y1, z1) = bounds

    X = np.linspace(x0, x1, nx)
    Y = np.linspace(y0, y1, ny)
    Z = np.array([z_level])
    dx = X[1] - X[0]

    P = _cartesian_product(X, Y, Z).astype(np.float32)
    P = torch.from_numpy(P).to(device)

    P.requires_grad_(True)
    y = model.forward(P)
    p = gradient(y, P)
    with torch.no_grad():
        norm_p = torch.linalg.norm(p, dim=1)
        residual = (norm_p - 1.0)**2
        pdf = torch.log(residual / epsilon)
        mark = pdf < 0
        pdf[mark] = 0
        pdf = pdf / (pdf.sum() * dx * dx)
        pdf = pdf.cpu().detach().numpy().reshape(nx, ny)
        
        fig1, ax = plt.subplots()
        plt.pcolormesh(pdf, cmap="coolwarm")
        plt.colorbar()
        ax.set_title(title + " SDF\n(min=%.6f max=%.6f)" % (np.min(pdf), np.max(pdf)))
        ax.tick_params(axis="both", which="major", labelsize=6)
        plt.show()

def plot_model_weight(model, writer=None):
    for key, val in model.model.state_dict().items():
        # print(key, val.max(), val.min(), val.shape)
        d = val.cpu().numpy()
        # print(key, np.min(d), np.max(d))
        if len(d.shape) == 1:
            d = d.reshape(-1, 1)
        fig, ax = plt.subplots()
        plt.imshow(d, cmap="coolwarm")
        ax.set_title("%s\nmin=%.6f\nmax=%.6f" % (key, np.min(d), np.max(d)))
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.set_xticks(np.arange(d.shape[1]))
        ax.set_yticks(np.arange(d.shape[0]))
        ax.set_xticklabels([x for x in range(d.shape[1])])
        ax.set_yticklabels([x for x in range(d.shape[0])])
        fig.tight_layout()
        plt.colorbar()
        plt.show()
