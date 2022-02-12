# -*- coding: utf-8 -*-

import os

import numpy as np
import sdf
from torch import from_numpy
from torch.utils.data import Dataset

from .geometry import (SDF, ImportanceImplicitSampler, ImportanceSampler, Mesh,
                       PointSampler)
from .pyigl_import import igl


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
    def __init__(
        self, f, N=1000, step=0.01, offset=10,
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
            dx, dy, dz = (step,)*3
            (x0, y0, z0), (x1, y1, z1) = sdf.mesh._estimate_bounds(f)
            X = np.linspace(x0-offset*dx, x1+offset*dx, _ndim, dtype=np.float32)
            Y = np.linspace(y0-offset*dy, y1+offset*dy, _ndim, dtype=np.float32)
            Z = np.linspace(y0-offset*dy, y1+offset*dy, _ndim, dtype=np.float32)
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
                sdf.write_binary_stl(temp_filename, f.generate(step, verbose=False, method=1))
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

class ResidualDataset(Dataset):
    def __init__(self, points=None, device=None):
        super().__init__()
        assert(points is not None)
        assert(points.shape[0] > 0)
        assert(points.shape[1] == 3)
        if device is not None:
            points = from_numpy(points.astype(np.float32)).to(device)
        
        self.points = points
    
    def __str__(self):
        return f'ResidualDataset ({self.points.shape[0]} points)'

    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx]}

class BoundaryConditionDataset(Dataset):
    def __init__(self, points=None, values=None, device=None):
        super().__init__()
        assert(points is not None)
        assert(points.shape[0] > 0)
        assert(points.shape[1] == 3)

        assert(values is not None)
        assert(values.shape[0] > 0)

        if len(values.shape) > 1:
            values = values.squeeze()

        if device is not None:
            points = from_numpy(points.astype(np.float32)).to(device)
            values = from_numpy(values.astype(np.float32)).to(device)
            
        self.points = points
        self.values = values
    
    def __str__(self):
        return f'BoundaryConditionDataset ({self.points.shape[0]} points)'

    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "bc": self.values[idx]}

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
    def __init__(self, mesh_file, N=10000, sampling_method='importance',
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
            else:
                raise ValueError('sampling_method must be \'point\' or \'importance\'')
            self.points = _sampler.sample(N).astype(np.float32)
            self.sdfs = _SDF.query(self.points).astype(np.float32)
        
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
    def __init__(self, mesh_file, N=1000000, offset=0, from_file=None, from_dataset=None, device=None):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
            self.gradients = dataset['gradients']
        elif from_dataset is not None:
            self.points = from_dataset['points']
            self.sdfs = from_dataset['sdfs']
            self.gradients = from_dataset['gradients']
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

            X = np.linspace(x0 - offset, x1 + offset, N, dtype=np.float32)
            Y = np.linspace(y0 - offset, y1 + offset, N, dtype=np.float32)
            Z = np.linspace(z0 - offset, z1 + offset, N, dtype=np.float32)

            self.points = sdf.mesh._cartesian_product(X, Y, Z)
            self.sdfs, _, _ = igl.signed_distance(self.points, v, f, 4, return_normals=False)
            self.gradients = np.linalg.norm(
                np.gradient(self.sdfs.reshape((N,N,N)), dx, dy, dz),
                axis=0
            ).reshape((N*N*N,))
        
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
    
    @classmethod
    def from_dataset(cls, dataset=None, device=None):
        if dataset is None:
            raise ValueError('dataset is None')
        return cls(None, from_dataset=dataset, device=device)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx], "gradient": self.gradients[idx]}

    def __str__(self):
        return 'UniformMeshSDFDataset (%d points)' % self.sdfs.shape[0]

class OutsideMeshSDFDataset(Dataset):
    """
    Init dataset of outside points (SDF > 0 for IGL.sign_distance or SDF < 0 for FREP)
    random uniform points from the bounding box * scale

    Parameters
    ----------
    mesh_file: str
        3D mesh file name
    N: int
        the number of samples (default: 10000)
    scale: float
        the number used to increase boundary of random point outside
    from_file: str
        npz path to load datset from file
    device: str
        torch device
    """
    def __init__(self, mesh_file, N=1000000, scale=2, from_file=None, from_dataset=None, device=None):
        super().__init__()
        
        if from_file is not None:
            dataset = np.load(from_file)
            self.points = dataset['points']
            self.sdfs = dataset['sdfs']
        elif from_dataset is not None:
            self.points = from_dataset['outside_points']
            self.sdfs = from_dataset['outside_sdfs']
        else:
            if not os.path.exists(mesh_file):
                raise ValueError(f'{mesh_file} did not exists')
            
            # Load mesh
            v,f = igl.read_triangle_mesh(mesh_file)
            
            # Get bounding box
            bv, bf = igl.bounding_box(v)
            (x0, y0, z0), (x1, y1, z1) = bv[0]*scale, bv[-1]*scale

            X = np.random.uniform(x0, x1, int(N*scale))
            Y = np.random.uniform(y0, y1, int(N*scale))
            Z = np.random.uniform(z0, z1, int(N*scale))

            points = np.vstack((X,Y,Z)).T

            sdfs, _, _ = igl.signed_distance(points, v, f, 4, return_normals=False)
            mark = sdfs > 0
            self.sdfs = sdfs[mark]
            self.points = points[mark]
            if mark.sum() > N:
                mark = np.random.choice(self.points.shape[0], N, replace=False)
                self.sdfs = self.sdfs[mark]
                self.points = self.points[mark]
        
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
        return 'OutsideMeshSDFDataset (%d points)' % self.sdfs.shape[0]

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
        self.outside = OutsideMeshSDFDataset.from_dataset(dataset, device=device)
    
    def __str__(self):
        return f'TestDataset ({len(self.uniform)} points, {len(self.random)} points, {len(self.outside)} points)'

def generate_dataset(f, name=None, N_train=1e4, N_test=1e5, N_slice=100, step=0.01, save_dir=os.getcwd(), verbose=True) -> None:
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
        N=int(N_train), step=step,
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
    test_random_importance_dataset = RandomMeshSDFDataset(output_stl, N=int(N_test), sampling_method='importance')

    if verbose:
        print(test_random_importance_dataset)

    # Generate uniform test datasets with offset
    test_outside_dataset = OutsideMeshSDFDataset(output_stl, N=int(N_test), scale=3)

    if verbose:
        print(test_outside_dataset)
    

    # Merge and save dataset to npz
    np.savez_compressed(
        output_test_npz,
        points=test_uniform_dataset.points,
        sdfs=test_uniform_dataset.sdfs,
        gradients=test_uniform_dataset.gradients,
        random_points=test_random_importance_dataset.points,
        random_sdfs=test_random_importance_dataset.sdfs,
        outside_points=test_outside_dataset.points,
        outside_sdfs=test_outside_dataset.sdfs
    )


def batch_loader(x, y=None, z=None, batch_size=None, num_batches=10):
    assert(hasattr(x, 'shape'))
    assert(len(x.shape) == 2)
    total_length = x.shape[0]
 
    if y is not None:
        assert(total_length == y.shape[0])
    
    if z is not None:
        assert(total_length == z.shape[0])

    if batch_size is None:
        batch_size = total_length // num_batches
    
    if y is not None:
        if z is not None:
            return (
                (
                    x[start:start+batch_size],
                    y[start:start+batch_size],
                    z[start:start+batch_size]
                ) 
                for start in range(0, total_length, batch_size)
            )
        return (
            (
                x[start:start+batch_size],
                y[start:start+batch_size]
            ) 
            for start in range(0, total_length, batch_size)
        )
    
    return (x[start:start+batch_size] for start in range(0, total_length, batch_size))
