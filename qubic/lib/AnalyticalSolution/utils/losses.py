import torch

def edge_mse_1d(pred, true, edge_index):
    """ MSE of differences across edges for 1D data """
    s, d = edge_index
    return ((pred[s]-pred[d]) - (true[s]-true[d])).pow(2).mean()

def edge_mse_iqu(pred, true, edge, weigh_QU=1.):
    """ MSE of differences across edges for 3D data (I, Q, U) with optional weighting."""
    return edge_mse_1d(pred[:,0], true[:,0], edge) + weigh_QU*edge_mse_1d(pred[:,1], true[:,1], edge) + weigh_QU*edge_mse_1d(pred[:,2], true[:,2], edge)