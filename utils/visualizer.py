# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from matplotlib.colors import CenteredNorm
from sdf.mesh import _cartesian_product, _estimate_bounds
from utils.dataset import SliceDataset
from utils.helpers import gradient


class SDFVisualize:
    def __init__(self, z_level=0, step=0.01, offset=30, nums=100, tag=None, writer=None, device="cpu"):
        self.z_level = z_level
        self.offset = offset
        self.nums = nums
        self.step = step
        self.tag = tag if tag is not None else str(int(time.time()))
        self.writer = writer
        self.device = device

    def from_nn(self, model, bounds=[(-1, -1, -1), (1, 1, 1)], bounds_from_mesh=None, title="predict"):
        dx, dy, dz = (self.step,) * 3
        (x0, y0, z0), (x1, y1, z1) = (
            self._bounds_from_mesh(bounds_from_mesh) if not bounds_from_mesh is None else bounds
        )

        X = np.linspace(x0 - self.offset * dx, x1 + self.offset * dx, self.nums)
        Y = np.linspace(y0 - self.offset * dy, y1 + self.offset * dy, self.nums)
        Z = np.array([self.z_level])

        P = _cartesian_product(X, Y, Z).astype(np.float32)
        P = torch.from_numpy(P).to(self.device)
        P.requires_grad = True
        sdf = model.forward(P)

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

        from utils.libs import igl

        v, f = igl.read_triangle_mesh(file)
        bv, bf = igl.bounding_box(v)

        dx, dy, dz = (self.step,) * 3
        (x0, y0, z0), (x1, y1, z1) = bv[0], bv[-1]

        X = np.linspace(x0 - self.offset * dx, x1 + self.offset * dx, self.nums)
        Y = np.linspace(y0 - self.offset * dy, y1 + self.offset * dy, self.nums)
        Z = np.array([self.z_level])

        P = _cartesian_product(X, Y, Z)

        sdf, _, _ = igl.signed_distance(P, v, f, sign_type, return_normals=False)
        sdf = sdf.reshape((self.nums, self.nums))

        gradient = np.linalg.norm(np.array(np.gradient(sdf, X[1] - X[0], Y[1] - Y[0])), axis=0)

        gradient_2 = np.linalg.norm(np.array(np.gradient(gradient, X[1] - X[0], Y[1] - Y[0])), axis=0)

        if self.writer is not None:
            self.writer.add_figure(self.tag + "/true_sdf", self._plot(sdf, gradient, gradient_2), close=True)
        else:
            self._plot(sdf, gradient, title=title)

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

    def _plot(self, sdf, grad=None, grad2=None, title=""):
        figs = []
        if self.writer is not None:
            plt.switch_backend("agg")

        fig1, ax = plt.subplots()
        plt.pcolormesh(sdf, cmap="coolwarm", norm=CenteredNorm())
        plt.colorbar()
        ax.set_title(title + " SDF\n(min=%.6f max=%.6f)" % (np.min(sdf), np.max(sdf)))
        ax.tick_params(axis="both", which="major", labelsize=6)
        if self.writer is None:
            plt.show()
        else:
            figs.append(fig1)

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

        return figs

    def _bounds_from_mesh(self, file):
        from utils.libs import igl

        v, _ = igl.read_triangle_mesh(file)
        bv, _ = igl.bounding_box(v)
        return (bv[0], bv[-1])


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
