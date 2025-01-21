import sys
import os
import qubic
import numpy as np
import component_separation as compsep

QUBIC_DATADIR = os.environ['QUBIC_DATADIR']

d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
d150.read_from_file(QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format('FI-150'))
d220.read_from_file(QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format('FI-220'))
coverage = compsep.get_coverage_from_file()

for seed in np.random.randint(low=10, high=1000000, size=10):
    for n in [3, 4, 5]:
        d150['nf_recon'] = d220['nf_recon'] = n  # nbr of reconstructed bands (output)
        d150['nf_sub'] = d220['nf_sub'] = 4 * n  # nbr of simulated bands (input)
        for ny in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]:
            kw_args = {'noise_profile': False,
                       'nunu': False,
                       'sc': False,
                       'seed': seed,
                       'save_maps': True,
                       'return_maps': False}
            compsep.generate_cmb_dust_maps(d150, coverage, ny, **kw_args)
            compsep.generate_cmb_dust_maps(d220, coverage, ny, **kw_args)

sys.exit(0)
