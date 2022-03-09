from utils.dataset_generator import run_batch
from kaolin.metrics import voxelgrid
import math

def test_iou(net, x, true_sdf, batch_size=10000, eps = 0.00001):
    assert hasattr(net, 'predict'), 'nn.Module must has predict function, i.e. extending from Base'
    # predict sdf from net
    predict_sdf = run_batch(net.predict, x, batch_size=batch_size)
    # convert list to voxel grid
    N = x.shape[0]
    Nx = math.ceil(N**(1/3))
    
    # threshould the sdf into a binary voxelgrid
    _mark = true_sdf > eps
    true_sdf[_mark] = 0.0
    true_sdf[~_mark] = 1.0

    _mark = predict_sdf > eps
    predict_sdf[_mark] = 0.0
    predict_sdf[~_mark] = 1.0

    voxelgrid_ground = true_sdf.reshape((1, Nx, Nx, Nx))
    voxelgrid_pred = predict_sdf.reshape((1, Nx, Nx, Nx))
    return voxelgrid.iou(voxelgrid_pred, voxelgrid_ground)