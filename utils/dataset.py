# -*- coding: utf-8 -*-

import sdf
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np
from .libs import igl
from .geometry import Mesh, SDF, PointSampler, ImportanceSampler, ImportanceImplicitSampler

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
    def __init__(
        self, f, N=1000, step=0.01, offset=10, atol=0.01, beta=10,
        output_stl='tmp.stl', device=None, from_file:str=None, from_dataset:dict=None,
        *args, **vargs
    ):
        super().__init__()

        # load dataset from npz
        if from_file is not None:
            dataset = np.load(from_file)
            self.pde_points = dataset['pde_points']
            self.bc_points = dataset['bc_points']
            self.bc_sdfs = dataset['bc_sdfs']
        elif from_dataset is not None:
            self.pde_points = from_dataset['pde_points']
            self.bc_points = from_dataset['bc_points']
            self.bc_sdfs = from_dataset['bc_sdfs']
        # otherwise generate from SDF
        else:
            if not isinstance(f, sdf.SDF3):
                raise TypeError('Cannot init ImplicitDataset(f) because f is not sdf.SDF3 object')
            

            _ndim = round((10*N)**(1/3))
            if _ndim <= 0:
                raise ValueError('N must be greater than or equal to 1')

            # Generate grid (x,y,z) from cartesian product
            dx, dy, dz = (step,)*3
            (x0, y0, z0), (x1, y1, z1) = sdf.mesh._estimate_bounds(f)
            X = np.linspace(x0-offset*dx, x1+offset*dx, _ndim, dtype=np.float32)
            Y = np.linspace(y0-offset*dy, y1+offset*dy, _ndim, dtype=np.float32)
            Z = np.linspace(y0-offset*dy, y1+offset*dy, _ndim, dtype=np.float32)
            P = sdf.mesh._cartesian_product(X, Y, Z)

            # Filter out |F(x)| > eps and set these points as Dirichlet boundary condition
            lvl_set = f(P)
            _mark = np.squeeze(np.isclose(lvl_set, np.zeros_like(lvl_set), atol=atol))
            # Filter out |F(x)| < max - eps
            #_max_sdf = np.max(lvl_set)
            #_max_mark = np.squeeze(np.isclose(_max_sdf - lvl_set, np.zeros_like(lvl_set), atol=atol))
            self.bc_points = P[_mark]
            self.bc_sdfs = lvl_set[_mark].squeeze()

            sampler = ImportanceImplicitSampler(lvl_set, beta=beta)
            _mark = sampler.sample(N)
            self.pde_points = P[_mark, :]

            try:
                temp_filename = os.path.normpath(output_stl)
                sdf.write_binary_stl(temp_filename, f.generate(step, verbose=False, method=1))
            #     # calculate SDF of bc_points
            #     _mesh = geometry.Mesh(temp_filename, doNormalize=True)
            #     _SDF = geometry.SDF(_mesh)
            #     self.bc_sdfs = _SDF.query(self.bc_points)
            #     # sampling 3.5x of bc_points for residual points
            #     _sampler = geometry.PointSampler(_mesh, ratio=1.0)
            #     self.pde_points = _sampler.sample(math.ceil(3.5*len(self.bc_points)))
            except Exception as e:
                print("warning: ", e)

        if device is not None:
            self.pde_points = from_numpy(self.pde_points.astype(np.float32)).to(device)
            self.bc_points = from_numpy(self.bc_points.astype(np.float32)).to(device)
            self.bc_sdfs = from_numpy(self.bc_sdfs.astype(np.float32)).to(device)

    @classmethod
    def to_file(cls, *arg, **varg):
        if 'file' not in varg:
            raise ValueError('file is None')
        file = varg['file']
        del varg['file']
        
        dataset = cls(*arg, **varg)
        np.savez_compressed(
            file,
            pde_points=dataset.residual.points,
            bc_points=dataset.bc.points,
            bc_sdfs=dataset.bc.values
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
        return 'ImplicitDataset (%d PDE points, %d Dirichlet BCs)' % (len(self.pde_points), len(self.bc_points))

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
    def __init__(self, mesh_file, N=10000, sampling_method='uniform',
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
    def __init__(self, mesh_file, N=1000000, from_file=None, from_dataset=None, device=None):
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

            X = np.linspace(x0, x1, N, dtype=np.float32)
            Y = np.linspace(y0, y1, N, dtype=np.float32)
            Z = np.linspace(z0, z1, N, dtype=np.float32)

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

    # Merge and save dataset to npz
    np.savez_compressed(
        output_test_npz,
        points=test_uniform_dataset.points,
        sdfs=test_uniform_dataset.sdfs,
        gradients=test_uniform_dataset.gradients,
        random_points=test_random_importance_dataset.points,
        random_sdfs=test_random_importance_dataset.sdfs
    )