# -*- coding: utf-8 -*-

import sdf
from weight_encode_neural_implicit import geometry
from torch.utils.data import Dataset
import numpy as np

import os

class ImplicitDataSet(Dataset):
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
    def __init__(self, f, N=1000, step=0.01, offset=30, atol=2e-1, output_stl='tmp.stl', sampling_method='point', *args, **vargs):
        if not isinstance(f, sdf.SDF3):
            raise TypeError('Cannot init ImplicitDataset(f) because f is not sdf.SDF3 object')
        super().__init__()

        _ndim = round(N**(1/3))
        if _ndim <= 0:
            raise ValueError('N must be greater than or equal to 1')

        dx, dy, dz = (step,)*3
        (x0, y0, z0), (x1, y1, z1) = sdf.mesh._estimate_bounds(f)
        X = np.linspace(x0-offset*dx, x1+offset*dx, _ndim)
        Y = np.linspace(y0-offset*dy, y1+offset*dy, _ndim)
        Z = np.linspace(y0-offset*dy, y1+offset*dy, _ndim)
        P = sdf.mesh._cartesian_product(X, Y, Z)

        lvl_set = f(P)
        _zero_mark = np.squeeze(np.isclose(lvl_set, np.zeros_like(lvl_set), atol=atol))
        self.points = P[_zero_mark]

        try:
            temp_filename = os.path.normpath(output_stl)
            sdf.write_binary_stl(temp_filename, f.generate(step, verbose=False, method=1))
            _mesh = geometry.Mesh(temp_filename, doNormalize=True)
            _SDF = geometry.SDF(_mesh)
            self.sdfs = _SDF.query(self.points)
        except Exception as e:
            print("warning: ", e)
        
    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx]}

class MeshSDFDataSet(Dataset):
    """
    Init dataset with SDF of Mesh

    Parameters
    ----------
    f: sdf.SDF3
        implicit function
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
    def __init__(self, file, N=10000, sampling_method='point', *args, **vargs):
        super().__init__()
        
        if not os.path.exists(file):
            raise ValueError(f'{file} did not exists')

        _mesh = geometry.Mesh(file, doNormalize=True)
        _SDF = geometry.SDF(_mesh)
        
        if sampling_method == 'point':
            _sampler = geometry.PointSampler(_mesh, *args, **vargs)
        elif sampling_method == 'importance':
            _sampler = geometry.ImportanceSampler(_mesh, *args, **vargs)
        else:
            raise ValueError('sampling_method must be \'point\' or \'importance\'')
        
        self.points = _sampler.sample(N)
        self.sdfs = _SDF.query(self.points)

    def __len__(self):
        return self.sdfs.shape[0]
    
    def __getitem__(self, idx):
        return {"x": self.points[idx], "sdf": self.sdfs[idx]}