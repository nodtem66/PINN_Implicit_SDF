# -*- coding: utf-8 -*-

from . import geometry
import sdf
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np
import math
from .libs import igl

import os

class ImplicitDataset():
    """
    Init dataset with implicit function f

    Parameters
    ----------
    f: sdf.SDF3
        implicit function
    N: int
        the number of samples (default: 1000)
    sampling_method: str
        'point' for point sampling
        'importance' for importance sampling
        see weight_encode_neural_implicit/geometry.py
    *args
        These parameters will be passed to the sampling class
    **vargs
        These parameters will be passed to the sampling class
    """
    def __init__(self, f, N=1000, step=0.01, offset=30, atol=2e-1, output_stl='tmp.stl', sampling_method='point', device=None, from_file=None, *args, **vargs):
        super().__init__()

        # load dataset from npz
        if from_file is not None:
            dataset = np.load(from_file)
            self.pde_points = dataset['pde_points']
            self.bc_points = dataset['bc_points']
            self.bc_sdfs = dataset['bc_sdfs']
        # otherwise generate from SDF
        else:
            if not isinstance(f, sdf.SDF3):
                raise TypeError('Cannot init ImplicitDataset(f) because f is not sdf.SDF3 object')
            

            _ndim = round(N**(1/3))
            if _ndim <= 0:
                raise ValueError('N must be greater than or equal to 1')

            # Generate grid (x,y,z) from cartesian product
            dx, dy, dz = (step,)*3
            (x0, y0, z0), (x1, y1, z1) = sdf.mesh._estimate_bounds(f)
            X = np.linspace(x0-offset*dx, x1+offset*dx, _ndim)
            Y = np.linspace(y0-offset*dy, y1+offset*dy, _ndim)
            Z = np.linspace(y0-offset*dy, y1+offset*dy, _ndim)
            P = sdf.mesh._cartesian_product(X, Y, Z)

            # Filter out |F(x)| > eps and set these points as Dirichlet boundary condition
            lvl_set = f(P)
            _zero_mark = np.squeeze(np.isclose(lvl_set, np.zeros_like(lvl_set), atol=atol))
            # Filter out |F(x)| < max - eps
            _max_sdf = np.max(lvl_set)
            _max_mark = np.squeeze(np.isclose(_max_sdf - lvl_set, np.zeros_like(lvl_set), atol=atol))
            self.bc_points = P[_zero_mark | _max_mark]

            try:
                temp_filename = os.path.normpath(output_stl)
                sdf.write_binary_stl(temp_filename, f.generate(step, verbose=False, method=1))
                # calculate SDF of bc_points
                _mesh = geometry.Mesh(temp_filename, doNormalize=True)
                _SDF = geometry.SDF(_mesh)
                self.bc_sdfs = _SDF.query(self.bc_points)
                # sampling 3.5x of bc_points for residual points
                _sampler = geometry.PointSampler(_mesh, ratio=1.0)
                self.pde_points = _sampler.sample(math.ceil(3.5*len(self.bc_points)))
            except Exception as e:
                print("warning: ", e)

        if device is not None:
            self.pde_points = from_numpy(self.pde_points).to(device)
            self.bc_points = from_numpy(self.bc_points).to(device)
            self.bc_sdfs = from_numpy(self.bc_sdfs).to(device)

    @classmethod
    def to_file(cls, *arg, **varg):
        if 'file' not in varg:
            raise ValueError('file is None')
        file = varg['file']
        del varg['file']
        
        dataset = cls(*arg, **varg)
        np.savez_compressed(
            file,
            pde_points=dataset.pde_points,
            bc_points=dataset.bc_points,
            bc_sdfs=dataset.bc_sdfs
        )
        return dataset

    @classmethod
    def from_file(cls, file=None, device=None):
        if file is None:
            raise ValueError('file is None')
        return cls(None, from_file=file, device=device)

    def __str__(self):
        return 'Implicit dataset (%d PDE points, %d Dirichlet BCs)' % (self.pde_points.shape[0], self.bc_points.shape[0])

class RandomMeshSDFDataset(Dataset):
    """
    Init dataset with SDF of Mesh

    Parameters
    ----------
    mesh_file: str
        path of mesh file
    N: int
        the number of samples (default: 10000)
    sampling_method: str
        'point' for point sampling
        'importance' for importance sampling
        see weight_encode_neural_implicit/geometry.py
    *args
        These parameters will be passed to the sampling class
    **vargs
        These parameters will be passed to the sampling class
    """
    def __init__(self, mesh_file, N=10000, sampling_method='uniform', from_file=None, device=None, *args, **vargs):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            _mesh = geometry.Mesh(mesh_file, doNormalize=True)
            _SDF = geometry.SDF(_mesh)
            
            if sampling_method == 'uniform':
                ratio = vargs.pop('ratio', 1.0)
                _sampler = geometry.PointSampler(_mesh, ratio=ratio)
            elif sampling_method == 'point':
                _sampler = geometry.PointSampler(_mesh, *args, **vargs)
            elif sampling_method == 'importance':
                _sampler = geometry.ImportanceSampler(_mesh, M=10*N, W=10)
            else:
                raise ValueError('sampling_method must be \'point\' or \'importance\'')
            self.points = _sampler.sample(N)
            self.sdfs = _SDF.query(self.points)
        
        if device is not None:
            self.points = from_numpy(self.points).to(device)
            self.sdfs = from_numpy(self.sdfs).to(device)

    
    @classmethod
    def to_file(cls, *arg, **kwarg):
        if 'file' not in kwarg:
            raise ValueError('file is None')
        file = kwarg['file']
        del kwarg['file']

        dataset = cls(*arg, **kwarg)
        np.savez_compressed(
            file,
            points=dataset.points,
            sdfs=dataset.sdfs
        )
        return dataset

    @classmethod
    def from_file(cls, file=None, device=None):
        if file is None:
            raise ValueError('file is None')
        return cls(None, from_file=file, device=device)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx]}

    def __str__(self):
        return 'RandomMeshSDFDataset (%d points)' % self.sdfs.shape[0]

class UniformMeshSDFDataset(Dataset):
    """
    Init dataset with SDF of Mesh and calculate the norm of gradient for testing

    Parameters
    ----------
    mesh_file: str
        3D mesh file name
    N: int
        the number of samples (default: 10000)
    from_file: str
        npz path to load datset from file
    device: str
        torch device
    """
    def __init__(self, mesh_file, N=1000000, from_file=None, device=None):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
            self.gradients = dataset['gradients']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            
            N = round(N**(1/3))
            if N <= 0:
                raise ValueError('N must be greater than or equal to 1')
            
            # Load mesh
            v,f = igl.read_triangle_mesh(mesh_file)
            
            # Get bounding box
            bv, bf = igl.bounding_box(v)
            (x0, y0, z0), (x1, y1, z1) = bv[0], bv[-1]

            # Calculate step
            dx, dy, dz = np.abs(bv[0]-bv[-1]) / N

            X = np.linspace(x0, x1, N)
            Y = np.linspace(y0, y1, N)
            Z = np.linspace(z0, z1, N)

            self.points = sdf.mesh._cartesian_product(X, Y, Z)
            self.sdfs, _, _ = igl.signed_distance(self.points, v, f, 4, return_normals=False)
            self.gradients = np.linalg.norm(
                np.gradient(self.sdfs.reshape((N,N,N)), dx, dy, dz),
                axis=0
            )
        
        if device is not None:
            self.points = from_numpy(self.points).to(device)
            self.sdfs = from_numpy(self.sdfs).to(device)
            self.gradients = from_numpy(self.gradients).to(device)

    @classmethod
    def to_file(cls, *arg, **kwarg):
        if 'file' not in kwarg:
            raise ValueError('file is None')
        file = kwarg['file']
        del kwarg['file']

        dataset = cls(*arg, **kwarg)
        np.savez_compressed(
            file,
            points=dataset.points,
            sdfs=dataset.sdfs,
            gradients=dataset.gradients
        )
        return dataset

    @classmethod
    def from_file(cls, file=None, device=None):
        if file is None:
            raise ValueError('file is None')
        return cls(None, from_file=file, device=device)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx], "gradient": self.gradients[idx]}

    def __str__(self):
        return 'UniformMeshSDFDataset (%d points)' % self.sdfs.shape[0]

class TestDataset(Dataset):
    def __init__(self, npz_file, device=None):
        assert(os.path.exists(npz_file))
        dataset = np.load(npz_file)
        self.points = dataset['points']
        self.sdfs = dataset['sdfs']
        self.gradients = dataset['gradients']
        self.random_points = dataset['random_points']
        self.random_sdfs = dataset['random_sdfs']
    
    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {
            "x": self.points[idx], "sdf": self.sdfs[idx], "gradient": self.gradients[idx],
            "random_x": self.random_points[idx], "random_sdf": self.random_sdfs[idx]
        }

    def __str__(self):
        return 'TestDataset (%d points)' % self.sdfs.shape[0]

def generate_dataset(f, name=None, N_train=1000, N_test=1e6, step=0.01, save_dir=os.getcwd()) -> None:
    if name is None:
        name = 'tmp_' + os.urandom(6).hex()
    output_stl = os.path.join(save_dir, name + '.stl')
    output_train_npz = os.path.join(save_dir, name + '_train.npz')
    output_test_npz = os.path.join(save_dir, name + '_test.npz')
    print(f'Saved file at {output_stl}')
    
    # Generate train dataset
    train_dataset = ImplicitDataset.to_file(
        f, 
        N=int(N_train), step=step,
        file=output_train_npz, output_stl=output_stl
    )

    # Generate 2 test datasets
    test_uniform_dataset = UniformMeshSDFDataset(output_stl, N=int(N_test))
    test_random_importance_dataset = RandomMeshSDFDataset(output_stl, N=int(N_test), sampling_method='importance')
    np.savez_compressed(
        output_test_npz,
        points=test_uniform_dataset.points,
        sdfs=test_uniform_dataset.sdfs,
        gradients=test_uniform_dataset.gradients,
        random_points=test_random_importance_dataset.points,
        random_sdfs=test_random_importance_dataset.sdfs
    )