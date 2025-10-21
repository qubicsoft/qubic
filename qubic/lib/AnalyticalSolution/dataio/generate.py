import numpy as np
from tqdm import tqdm
import healpy as hp
import qubic.lib.QskySim as qss

from qubic.lib.AnalyticalSolution.operators import ForwardOps

def generate_sky_projectedsky_pairs(d, q, s, num_samples, acquisition, P_operator=None, seed = None, dust_model = 'd0'):
    """ Generate pairs of (sky, projected-sky) samples for training of the backprojection."""
    if seed is None:
        seed = np.random.randint(0, 100)
    sky_list, tod_list, map_list = [], [], []
    conv = acquisition.get_convolution_peak_operator()
    fops = ForwardOps(q, acquisition, s)
    if P_operator is None:
        P_operator = acquisition.get_projection_operator()
    for _ in tqdm(range(num_samples), desc="Generating sky–projected-sky pairs"):
        sky_cfg = {'cmb': seed, 'dust': dust_model}
        sky_map = qss.Qubic_sky(sky_cfg, d).get_simple_sky_map()[0]
        convolved = conv(sky_map)
        map_list.append(convolved)
        sky_preP = fops.op_filter()(fops.op_aperture_integration()(fops.op_unit_conversion()(convolved)))
        tod = P_operator(sky_preP)
        sky_list.append(sky_preP)
        tod_list.append(tod)
    return np.array(sky_list), np.array(tod_list), np.array(map_list)
