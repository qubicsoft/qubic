from qubic.scripts.Scanning_Strategy import Sky_Dips_Sim

import pickle

params_dict = {
        'latitude': -24.183333333,
        'longitude': -66.466666667,
        'RA_center': 245,
        'DEC_center': -61.45,
        'date_obs': '2026-01-01 00:00:00',
        'duration': 10,  # in hours
        'angspeed': 1,  # deg/s
        'delta_az': 20.0,  # deg
        'nsweeps_per_elevation': 25,
        'period': 1,  # s
        }

sampling = Sky_Dips_Sim.QubicObservation(params_dict).get_samplings()

with open('test_pointings.pkl', 'wb') as f:
    pickle.dump(sampling, f)
