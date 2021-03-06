# -*- coding: utf-8 -*-
# Test the python module installed in OS

# Add parent directory into system path
import sys
import os

def main():
    try:
        sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

        import numpy as np
        import torch
        import tensorboardX
        import tensorboard
        import sdf
        import PyScaffolder
        import matplotlib
        from tqdm import tqdm
        import dotenv
        import pretty_errors
        import skfmm
        import torchinfo
        import siren_pytorch

        from utils.external_import import igl
        print('Load all dependent modules: [OK]')
    except ImportError as e:
        print('Load all dependent modules: [Fail]')
        raise e

    try:
        sdf.sphere(1).generate(step=0.01, verbose=False, method=1)
        print('Test marching cubes: [OK]')
    except Exception as e:
        print('Test marching cubes: [Fail]')
        raise e

    try:
        v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]
        ]

        f = [
            [1, 2, 0],
            [2, 1, 3],
            [7, 2, 3],
            [2, 7, 6],
            [1, 7, 3],
            [7, 1, 5],
            [7, 4, 6],
            [4, 7, 5],
            [4, 2, 6],
            [2, 4, 0],
            [4, 1, 0],
            [1, 4, 5]
        ]

        from utils.geometry import SDF
        igl.signed_distance(np.array([[0.0, 0.0, 0.0], [.5, .5, .5], [-.5, -.5, -.5]]),
                            np.array(v), np.array(f), SDF.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, False)
        print('Test pyigl SDF calculation: [OK]')
    except Exception as e:
        print('Test pyigl SDF calculation: [Fail]')
        raise e

    try:
        from kaolin.metrics.pointcloud import chamfer_distance, f_score
        p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
                            [8.5640, 7.7767, 9.4214]],
                            [[0.5431, 6.4495, 11.4914],
                            [3.2126, 8.0865, 3.1018]]], device='cuda', dtype=torch.float)
        p2 = torch.tensor([[[6.9340, 6.1152, 3.4435],
                            [0.1032, 9.8181, 11.3350]],
                            [[11.4006, 2.2154, 7.9589],
                            [4.2586, 1.4133, 7.2606]]], device='cuda', dtype=torch.float)
        chamfer_distance(p1, p2)
        f_score(p1, p2, radius=0.5)

        from kaolin.metrics.voxelgrid import iou
        pred = torch.tensor([[[[0., 0.],
                            [1., 1.]],
                            [[1., 1.],
                            [1., 1.]]]], device='cuda')
        gt = torch.ones((1,2,2,2), device='cuda')
        iou(pred, gt)
        print('Test kaolin metrics: [OK]')
    except Exception as e:
        print('Test kaolin metrics: [Fail]')
        raise e

if __name__ == '__main__':
    main()