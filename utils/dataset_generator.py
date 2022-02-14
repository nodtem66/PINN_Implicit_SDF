# -*- coding: utf-8 -*-

import os
import math

import torch
import numpy as np
import sdf
from scipy.stats import qmc
from torch import from_numpy
from torch.utils.data import Dataset

from .geometry import (SDF, ImportanceSampler, Mesh,
                       PointSampler, get_bounding_box_and_offset)
from .external_import import igl


class ImplicitDataset(Dataset):
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
    def __init__(
        self, f, N=1000, scale_offset=0.1,
        output_stl='tmp.stl', device=None, from_file:str=None, from_dataset:dict=None,
        *args, **vargs
    ):
        super().__init__()

        # load dataset from npz
        if from_file is not None or from_dataset is not None:
            dataset = np.load(from_file) if from_file is not None else from_dataset
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
            self.grads = dataset['grads']
            self.true_sdfs = dataset['true_sdfs']
            self.true_grads = dataset['true_grads']
        # otherwise generate from SDF
        else:
            if not isinstance(f, sdf.SDF3):
                raise TypeError('Cannot init ImplicitDataset(f) because f is not sdf.SDF3 object')

            _ndim = round(N**(1/3))
            if _ndim <= 0:
                raise ValueError('N must be greater than or equal to 1')

            # Generate grid (x,y,z) from cartesian product
            (x0, y0, z0), (x1, y1, z1) = sdf.mesh._estimate_bounds(f)
            offset_x = abs(scale_offset*(x0-x1))
            offset_y = abs(scale_offset*(y0-y1))
            offset_z = abs(scale_offset*(z0-z1))
            assert(x0 <= x1 and y0 <= y1 and z0 <= z1 and offset_x >= 0 and offset_y >= 0 and offset_z >= 0)

            X = np.linspace(x0-offset_x, x1+offset_x, _ndim, dtype=np.float32)
            Y = np.linspace(y0-offset_y, y1+offset_y, _ndim, dtype=np.float32)
            Z = np.linspace(y0-offset_z, y1+offset_z, _ndim, dtype=np.float32)
            self.points = sdf.mesh._cartesian_product(X, Y, Z)
            
            lvl_set = f(self.points).reshape((_ndim, _ndim, _ndim))
            dx = (X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0])
            import skfmm
            sdfs = skfmm.distance(lvl_set, dx=dx, periodic=[True, True, True])
            grad = np.array(np.gradient(sdfs, *dx))
            self.sdfs = sdfs.reshape((_ndim**3,)).squeeze()
            self.grads = grad.reshape((3, _ndim**3)).T

            try:
                temp_filename = os.path.normpath(output_stl)
                sdf.write_binary_stl(temp_filename, f.generate(step=dx[0], verbose=False, method=1))
                # Load mesh
                v,f = igl.read_triangle_mesh(temp_filename)
                self.true_sdfs, _, _ = igl.signed_distance(self.points, v, f, 4, return_normals=False)
                self.true_grads = np.array(np.gradient(self.true_sdfs.reshape((_ndim, _ndim, _ndim)), *dx))
                self.true_grads = self.true_grads.reshape((3, _ndim**3)).T

            except Exception as e:
                print("warning: ", e)

        if device is not None:
            self.points = from_numpy(self.points.astype(np.float32)).to(device)
            self.sdfs = from_numpy(self.sdfs.astype(np.float32)).to(device)
            self.grads = from_numpy(self.grads.astype(np.float32)).to(device)
            self.true_sdfs = from_numpy(self.true_sdfs.astype(np.float32)).to(device)
            self.true_grads = from_numpy(self.true_grads.astype(np.float32)).to(device)

    @classmethod
    def to_file(cls, *arg, **varg):
        if 'file' not in varg:
            raise ValueError('file is None')
        file = varg['file']
        del varg['file']
        
        dataset = cls(*arg, **varg)
        np.savez_compressed(
            file,
            points=dataset.points,
            sdfs=dataset.sdfs,
            grads=dataset.grads,
            true_sdfs=dataset.true_sdfs,
            true_grads=dataset.true_grads,
        )
        return dataset

    @classmethod
    def from_file(cls, file=None, device=None, lazy_load=False):
        if file is None:
            raise ValueError('file is None')
        return cls(None, from_file=file, device=device, lazy_load=lazy_load)

    @classmethod
    def from_dataset(cls, dataset=None, device=None):
        if dataset is None:
            raise ValueError('dataset is None')
        return cls(None, from_dataset=dataset, device=device)

    def __str__(self):
        return 'ImplicitDataset (%d points)' % (len(self.points))

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {
            "x": self.points[idx],
            "sdf": self.sdfs[idx],
            "grad": self.grads[idx],
            "true_sdf": self.true_sdfs[idx],
            "true_grad": self.true_grads[idx]
        }

class RandomMeshSDFDataset(Dataset):
    """
    Init dataset with SDF of Mesh

    Parameters
    ----------
    `mesh_file`: str
        path of mesh file
    `N`: int
        the number of samples (default: 10000)
    `scale_offset`: float
        only used in `halton` and `sobol` to define the lower and upper boundary:
        `boundary = bounding box * scale_offset`
    `sampling_method`: str
        - 'uniform' for uniform sampling in sphere
        - 'point' for point sampling (with ratio between point and surface sampling)
        - 'importance' for importance sampling around the surface
        - 'halton' for Halton sequence
        - 'sobol' for Sobol sequence
        see `utils/geometry.py`
    `*args`
        These parameters will be passed to the sampling class
    `**vargs`
        These parameters will be passed to the sampling class
    """
    def __init__(self, mesh_file, N=10000, scale_offset=1.0, sampling_method='importance',
        from_file=None, from_dataset=None, device=None,
        *args, **vargs
    ):

        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
        elif from_dataset is not None:
            self.points = from_dataset['random_points']
            self.sdfs = from_dataset['random_sdfs']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            
            _mesh = Mesh(mesh_file, doNormalize=True)
            _SDF = SDF(_mesh)
            
            if sampling_method == 'uniform':
                ratio = vargs.pop('ratio', 1.0)
                _sampler = PointSampler(_mesh, ratio=ratio)
            elif sampling_method == 'point':
                _sampler = PointSampler(_mesh, *args, **vargs)
            elif sampling_method == 'importance':
                _sampler = ImportanceSampler(_mesh, M=10*N, W=10)
            elif sampling_method == 'halton':
                _sampler = qmc.Halton(d=3, scramble=False)
            elif sampling_method == 'sobol':
                _sampler = qmc.Sobol(d=3, scramble=False)
            else:
                raise ValueError('sampling_method must be \'uniform\', \'point\', \'importance\', \'halton\', or \'sobol\'')
            
            if sampling_method in ('halton', 'sobol'):
                # Calculate next power 2
                m = math.ceil(math.log2(N))

                # Get bounding box
                bv, _ = _mesh.bounding_box()
                l_bounds = np.min(bv, axis=0)
                u_bounds = np.max(bv, axis=0)

                if not hasattr(scale_offset, '__len__'):
                    self.points = _sampler.random_base2(m=m)
                else:
                    self.points = np.empty((0,3), dtype=np.float32)
                    for _scale in scale_offset:
                        _points = _sampler.random(n=2**m)
                        self.points = np.vstack((self.points, _points))
                self.points = qmc.scale(self.points, l_bounds=l_bounds*scale_offset, u_bounds=u_bounds*scale_offset).astype(np.float32)
            else:
                self.points = _sampler.sample(N).astype(np.float32)
            self.sdfs = _SDF.query(self.points).astype(np.float32)
            self.sampling_method = sampling_method
        
        if device is not None:
            self.points = from_numpy(self.points.astype(np.float32)).to(device)
            self.sdfs = from_numpy(self.sdfs.astype(np.float32)).to(device)

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
    
    @classmethod
    def from_dataset(cls, dataset=None, device=None):
        if dataset is None:
            raise ValueError('dataset is None')
        return cls(None, from_dataset=dataset, device=device)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx]}

    def __str__(self):
        return f'RandomMeshSDFDataset[{self.sampling_method}] ({self.sdfs.shape[0]} points)'

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
    def __init__(self, mesh_file, N=1000000, scale_offset=0.0, from_file=None, from_dataset=None, device=None):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
            self.grads = dataset['grads']
            self.norm_grads = dataset['norm_grads']
        elif from_dataset is not None:
            self.points = from_dataset['points']
            self.sdfs = from_dataset['sdfs']
            self.grads = from_dataset['grads']
            self.norm_grads = from_dataset['norm_grads']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            
            N = round(N**(1/3))
            if N <= 0:
                raise ValueError('N must be greater than or equal to 1')
            
            # Load mesh
            v,f = igl.read_triangle_mesh(mesh_file)
            
            # Get bounding box
            (x0, y0, z0), (x1, y1, z1), (offset_x, offset_y, offset_z) = get_bounding_box_and_offset(v, scale_offset=scale_offset)
            assert(x0 <= x1 and y0 <= y1 and z0 <= z1 and offset_x >= 0 and offset_y >= 0 and offset_z >= 0)

            # Calculate step
            X = np.linspace(x0 - offset_x, x1 + offset_x, N, dtype=np.float32)
            Y = np.linspace(y0 - offset_y, y1 + offset_y, N, dtype=np.float32)
            Z = np.linspace(z0 - offset_z, z1 + offset_z, N, dtype=np.float32)
            dx = X[1] - X[0]
            dy = Y[1] - Y[0]
            dz = Z[1] - Z[0]

            self.points = sdf.mesh._cartesian_product(X, Y, Z)
            self.sdfs, _, _ = igl.signed_distance(self.points, v, f, 4, return_normals=False)
            self.grads = np.array(np.gradient(self.sdfs.reshape((N,N,N)), dx, dy, dz))
            self.norm_grads = np.linalg.norm(
                self.grads,
                axis=0
            ).reshape((N*N*N,))
            self.grads = self.grads.reshape((3, N**3)).T
        
        if device is not None:
            self.points = from_numpy(self.points.astype(np.float32)).to(device)
            self.sdfs = from_numpy(self.sdfs.astype(np.float32)).to(device)
            self.grads = from_numpy(self.grads.astype(np.float32)).to(device)
            self.norm_grads = from_numpy(self.norm_grads.astype(np.float32)).to(device)

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
            grads=dataset.grads,
            norm_grads=dataset.norm_grads,
        )
        return dataset

    @classmethod
    def from_file(cls, file=None, device=None):
        if file is None:
            raise ValueError('file is None')
        return cls(None, from_file=file, device=device)
    
    @classmethod
    def from_dataset(cls, dataset=None, device=None):
        if dataset is None:
            raise ValueError('dataset is None')
        return cls(None, from_dataset=dataset, device=device)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx], "gradient": self.grads[idx]}

    def __str__(self):
        return 'UniformMeshSDFDataset (%d points)' % self.sdfs.shape[0]

class SliceDataset(Dataset):
    def __init__(self, mesh_file, N=100, from_file=None, device=None):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
            self.gradients = dataset['gradients']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            
            if N <= 0:
                raise ValueError('N must be greater than or equal to 1')
            
            # Load mesh
            v,f = igl.read_triangle_mesh(mesh_file)
            
            # Get bounding box
            bv, bf = igl.bounding_box(v)
            (x0, y0, z0), (x1, y1, z1) = bv[0], bv[-1]

            # Calculate step
            dx, dy, dz = np.abs(bv[0]-bv[-1]) / N

            X = np.linspace(x0, x1, N, dtype=np.float32)
            Y = np.linspace(y0, y1, N, dtype=np.float32)
            Z = np.array([(z0+z1)/2], dtype=np.float32)

            self.points = sdf.mesh._cartesian_product(X, Y, Z)
            self.sdfs, _, _ = igl.signed_distance(self.points, v, f, 4, return_normals=False)
            self.gradients = np.linalg.norm(
                np.gradient(self.sdfs.reshape((N,N)), dx, dy),
                axis=0
            ).reshape((N*N,))
        
        if device is not None:
            self.points = from_numpy(self.points.astype(np.float32)).to(device)
            self.sdfs = from_numpy(self.sdfs.astype(np.float32)).to(device)
            self.gradients = from_numpy(self.gradients.astype(np.float32)).to(device)

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
    
    def __str__(self):
        return 'SliceDataset (%d points)' % self.sdfs.shape[0]

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx], "gradient": self.gradients[idx]}

class TestDataset():
    def __init__(self, npz_file, device=None):
        assert(os.path.exists(npz_file))
        dataset = np.load(npz_file)
        self.uniform = UniformMeshSDFDataset.from_dataset(dataset, device=device)
        self.random = RandomMeshSDFDataset.from_dataset(dataset, device=device)
    
    def __str__(self):
        return f'TestDataset ({len(self.uniform)} points, {len(self.random)} points)'

def generate_dataset(f, name=None, N_train=1e4, N_test=1e5, N_slice=100, save_dir=os.getcwd(), verbose=True) -> None:
    if name is None:
        name = 'tmp_' + os.urandom(6).hex()
    output_stl = os.path.join(save_dir, name + '.stl')
    output_train_npz = os.path.join(save_dir, name + '_train.npz')
    output_test_npz = os.path.join(save_dir, name + '_test.npz')
    output_slice_npz = os.path.join(save_dir, name + '_slice.npz')
    print(f'Saved file at {output_stl}')
    
    # Generate train dataset
    train_dataset = ImplicitDataset.to_file(
        f, 
        N=int(N_train), scale_offset=0.1,
        file=output_train_npz, output_stl=output_stl
    )

    if verbose:
        print(train_dataset)

    # Generate slice dataset
    slice_dataset = SliceDataset.to_file(output_stl, N=int(N_slice), file=output_slice_npz)

    if verbose:
        print(slice_dataset)

    # Generate uniform test datasets
    test_uniform_dataset = UniformMeshSDFDataset(output_stl, N=int(N_test))

    if verbose:
        print(test_uniform_dataset)
    
    # Generate random point test dataset
    # test_random_importance_dataset = RandomMeshSDFDataset(output_stl, N=int(N_test), sampling_method='importance')

    # if verbose:
    #     print(test_random_importance_dataset)

    # Generate QMC test dataset
    test_qmc_dataset = RandomMeshSDFDataset(output_stl, N=int(N_test), scale_offset=[1.0, 2.0, 3.0], sampling_method='sobol')
    if verbose:
        print(test_qmc_dataset)
    
    # Merge and save dataset to npz
    np.savez_compressed(
        output_test_npz,
        points=test_uniform_dataset.points,
        sdfs=test_uniform_dataset.sdfs,
        grads=test_uniform_dataset.grads,
        norm_grads=test_uniform_dataset.norm_grads,
        random_points=test_qmc_dataset.points,
        random_sdfs=test_qmc_dataset.sdfs
    )


def batch_loader(*args, batch_size=None, num_batches:int=10):
    """
    Fast batch loader without collate_fn
    
    Parameters
    ----------
    - :attr:`*args`: `torch.tensor` or `numpy.array` \n
        The first dimension (`shape[0]`) of all input must be the same
    - :attr:`batch_size`: int | None \n
        The number of batch size. If it equals to `None`, the `batch_size` will be calculated from `num_batches`
    - :attr:`num_batches`: int \n
        The total number of batches (default=10). If `batch_size` is not `None`, `num_batches` will be ignored.

    Returns
    -------
    Generator of tuples of a batch

    Example
    -------
    ```python
    >> bl = batch_loader(np.ones((100,3)), np.ones(100), batch_size=30)
    >> for x, y in bl:
    >>    print(x.shape, y.shape)
    (30, 3) (30,)
    (30, 3) (30,)
    (30, 3) (30,)
    (10, 3) (10,)
    ```
    """
    assert len(args) > 0, 'Missing input'
    assert all([hasattr(x, 'shape') for x in args]), 'arguments must be torch.tensor or numpy.array'
    total_length = [x.shape[0] for x in args]
    assert total_length.count(total_length[0]) == len(total_length), f'The first dimension of every tensor or array must be the same: {total_length}'
    
    if batch_size is None:
        batch_size = total_length[0] // num_batches

    return (
        tuple(x[start:start+batch_size] for x in args) for start in range(0, total_length[0], batch_size)
    )

def run_batch(callback, *args, reducer=None, **kwarg):
    """
    Run `callback` with `batch_loader`
    see `utils/dataset_generator.py`:`batch_loader` for more information

    Parameters
    ----------
    - :attr:`callback`: function to run for each batch
    - :attr:`*args`: `torch.tensor` or `numpy.array` \n
        The first dimension (`shape[0]`) of all input must be the same
    - :attr:`reducer`: function to combine the result of calculation for each batch
    - :attr:`batch_size`: int | None \n
        The number of batch size. If it equals to `None`, the `batch_size` will be calculated from `num_batches`
    - :attr:`num_batches`: int \n
        The total number of batches (default=10). If `batch_size` is not `None`, `num_batches` will be ignored.
    
    Returns:
    Any

    Example
    -------
    ```
    >> class A:
    >>    def f1(self, x, y):
    >>        return np.mean(np.mean(x) + y)
    >> a = A()
    >> run_batch(a.f1, np.ones((100, 3)), np.ones(100), reducer=np.mean, batch_size=30)
    2.0

    >> class B:
    >>    def f1(self, x, y):
    >>        return torch.mean(torch.mean(x) + y)
    >> b = B()
    >> run_batch(b.f1, torch.ones((100, 3), device='cuda'), torch.ones(100, device='cuda'), reducer=torch.mean, batch_size=30)
    tensor(2., device='cuda:0')
    ```
    """
    #is_self = 'self' in callback.__code__.co_varnames
    #callback_args_count = callback.__code__.co_argcount - (1 if is_self else 0) 
    #if callback_args_count != len(args):
    #    print(f'[warning] The number of arguments of callback have to match input arguments: {callback_args_count} != {len(args)}')
    
    result = [callback(*x) for x in batch_loader(*args, **kwarg)]
    if isinstance(args[0], torch.Tensor):
        assert hasattr(args[0], 'device'), 'torch.Tensor should have device attribute'
        result = torch.tensor(result, device=args[0].device)
    
    return result if reducer is None else reducer(result)