#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'Test the functions in the instrumentdb module.'

from __future__ import division

import unittest as ut

import healpy as hp
import numpy as np
import qubic
import pysm
from pysm.nominal import models


class TestSpectroImLib(ut.TestCase):

    def test_input_sky_pysm(self):

        d = qubic.qubicdict.qubicDict()
        d.read_from_file("parameters.dict")
        p = qubic.get_pointing(d)

        sky_config0 = {'cmb': models('c1', d['nside'])}
        sky0 = input_sky_pysm(sky_config0, d)
        self.assertTrue(sky0.shape[0] == int(d['nf_sub']))
        self.assertTrue(sky0.shape[1] == 12*d['nside']**2)
        self.assertTrue(sky0.shape[2] == 3)
        for j in range(3):
            for i in range(len(sky0)-1):
                self.assertTrue(np.all(sky0[i, :, j] == x0[i+1, :, j]))
        
        sky_config = {
            'synchrotron': models('s1', d['nside']),
            'dust': models('d1', d['nside']),
            'freefree': models('f1', d['nside']), #not polarized
            'cmb': models('c1', d['nside']),
            'ame': models('a1', d['nside']),  #not polarized
        }
        sky = input_sky_pysm(sky_config, d)
        
        self.assertTrue(np.allclose(expected, cond.matr))
