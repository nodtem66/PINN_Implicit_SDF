from utils.dataset_generator import run_batch
from kaolin.metrics import voxelgrid
import math

def test_iou(net, x, true_sdf, batch_size=10000):
    assert hasattr(net, 'predict'), 'nn.Module must has predict function, i.e. extending from Base'
    # predict sdf from net
    predict_sdf = run_batch(net.predict, x, batch_size=batch_size)
    # convert list to voxel grid
    N = x.shape[0]
    Nx = math.ceil(N**(1/3))
    voxelgrid_ground = true_sdf.reshape((1, Nx, Nx, Nx))
    voxelgrid_pred = predict_sdf.reshape((1, Nx, Nx, Nx))
    return voxelgrid.iou(voxelgrid_pred, voxelgrid_ground)