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
        import sdf
        import PyScaffolder
        import matplotlib
        import torch_optimizer
        from tqdm import tqdm
        import dotenv
        import pretty_errors

        from utils.libs import igl
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

if __name__ == '__main__':
    main()