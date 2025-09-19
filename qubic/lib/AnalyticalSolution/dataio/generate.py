import numpy as np
from tqdm import tqdm
import healpy as hp
import qubic.lib.QskySim as qss

from qubic.lib.AnalyticalSolution.operators import QubicForwardOps

def generate_sky_projectedsky_pairs(d, q, s, num_samples, acquisition, P_operator=None):
    """ Generate pairs of (sky, projected-sky) samples for training of the backprojection."""

    sky_list, tod_list = [], []
    conv = acquisition.get_convolution_peak_operator()
    fops = QubicForwardOps(q, acquisition, s)
    if P_operator is None:
        P_operator = acquisition.get_projection_operator()
    for _ in tqdm(range(num_samples), desc="Generating sky–projected-sky pairs"):
        sky_cfg = {'cmb': 42, 'dust': 'd0'}
        sky_map = qss.Qubic_sky(sky_cfg, d).get_simple_sky_map()[0]
        convolved = conv(sky_map)
        sky_preP = fops.op_filter(fops.op_aperture_integration(fops.op_unit_conversion(convolved)))
        tod = P_operator(sky_preP)
        sky_list.append(sky_preP)
        tod_list.append(tod)
    return np.array(sky_list), np.array(tod_list)
