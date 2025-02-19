# coding: utf-8
from astropy.io import fits
import sys,os
from configparser import ConfigParser

from pysimulators import Layout, LayoutGrid
from qubic.lib.Qhorns import HornLayout
from qubic.lib.Qutilities import find_file
from qubic.calfiles import PATH as cal_dir
import numpy as np

__all__ = ['QubicCalibration']

class QubicCalibration(object):
    """
    Class representing the QUBIC calibration tree. It stores the calibration
    file names and "hardcoded" values and provides access to them.
    If the path name of a calibration file is relative, it is first searched
    relatively to the working directory and if not found, in the calibration
    path.
    """
    def __init__(self, d, path=cal_dir):
        """
        Parameters
        ----------
        path : str, optional
            The directory path of the calibration tree. The default one is
            the one that is contained in the qubic package.
        detarray : str, optional
            The detector array layout calibration file name.
        hornarray : str, optional
            The horn array layout calibration file name.
        optics : str, optional
            The optics parameters calibration file name.
        primbeam : str, optional
            The primary beam parameter calibration file name.
        synthbeam : str, optional
            The synthetic beam parameter calibration file name.
        """
        if path is None:
            path = '.'
            
        self.path = os.path.abspath(path)

        # replace the wildcard with the configuration:  either TD or FI
        epsilon = 1.0e-9 # one Hz of margin for comparisons
        self.nu = int(d['filter_nu']/1e9)
        if self.nu>=130-epsilon and self.nu<=170+epsilon:
            nu_str = "150"
        elif self.nu>=190-epsilon and self.nu<=247.5+epsilon:
            nu_str = "220"
        else:
            nu_str = '%03i' % self.nu
        for key in ['detarray','hornarray','optics','primbeam','synthbeam']:
            calfile = d[key].replace('_CC','_%s' % d['config']).replace('_FFF','_%s' % nu_str)
            calfile_fullpath = find_file(os.path.join(self.path,calfile), verbosity=1)
            if calfile_fullpath is None:
                cmd = "self.%s = None" % key
            else:
                cmd = "self.%s = '%s'" % (key,calfile_fullpath)
            print('executing: %s' % cmd)
            exec(cmd)
        if d['debug']:
            print('self.synthbeam = %s' % self.synthbeam)

        

    def __str__(self):
        state = [('path', self.path),
                 ('detarray', self.detarray),
                 ('hornarray', self.hornarray),
                 ('optics', self.optics),
                 ('primbeam', self.primbeam),
                 ('synthbeam', self.synthbeam)]
        return '\n'.join([a + ': ' + repr(v) for a, v in state])

    __repr__ = __str__

    def get(self, name, *args):
        
        """
        Access calibration files.
        Parameters
        ----------
        name : str
            One of the following:
                - 'detarray'
                - 'hornarray'
                - 'optics'
                - 'primbeam'
        """
        
        if name == 'detarray':
            hdus = fits.open(self.detarray)
            version = hdus[0].header['format version']
            vertex = hdus[2].data
            frame = hdus[0].header['FRAME']
            if frame == 'ONAFP':
                # Make a pi/2 rotation from ONAFP -> GRF referential frame
                vertex[..., [0, 1]] = vertex[..., [1, 0]]
                vertex[..., 1] *= - 1
            shape = vertex.shape[:-2]
            removed = hdus[3].data.view(bool)
            ordering = hdus[4].data
            quadrant = hdus[5].data
            efficiency = hdus[6].data

            return shape, vertex, removed, ordering, quadrant, efficiency

        elif name == 'hornarray':
            hdus = fits.open(self.hornarray)
            version = hdus[0].header['format version']
            if version == '1.0':
                h = hdus[0].header
                spacing = h['spacing']
                center = hdus[1].data
                shape = center.shape[:-1]
                layout = Layout(shape, center=center, radius=h['innerrad'],
                                open=None)
                layout.spacing = spacing
            elif version == '2.0':
                h = hdus[0].header
                spacing = h['spacing']
                xreflection = h['xreflection']
                yreflection = h['yreflection']
                radius = h['radius']
                selection = ~hdus[1].data.view(bool)
                layout = LayoutGrid(
                    removed.shape, spacing, selection=selection, radius=radius,
                    xreflection=xreflection, yreflection=yreflection,
                    open=None)
            else:
                h = hdus[1].header
                spacing = h['spacing']
                xreflection = h['xreflection']
                yreflection = h['yreflection']
                angle = h['angle']
                radius = h['radius']
                selection = ~hdus[2].data.view(bool)
                shape = selection.shape
                layout = HornLayout(
                    shape, spacing, selection=selection, radius=radius,
                    xreflection=xreflection, yreflection=yreflection,
                    angle=angle, startswith1=True, id=None, open=None)
                layout.id = np.arange(len(layout))
            layout.center = np.concatenate(
                [layout.center, np.full_like(layout.center[..., :1], 0)], -1)
            layout.open = np.ones(len(layout), bool)
            return layout

        elif name == 'optics':
            dtype = [('name', 'S16'), ('temperature', float),
                     ('transmission', float), ('emissivity', float),
                     ('nstates_pol', int)]
            if self.optics.endswith('fits'):
                header = fits.open(self.optics)[0].header
                return {'focal length': header['flength'],
                        'detector efficiency': 1.,
                        'components': np.empty(0, dtype=dtype)}
            parser = ConfigParser()
            parser.read(self.optics)
            # ### The 2 next lines are commented as there is nothing in the section
            # ### "general" in the optics calibration file. Focal length has been moved to the dictionary.
            # keys = 'focal length',
            # out = dict((key, parser.getfloat('general', key)) for key in keys)
            out = {}
            raw = parser.items('components')
            components = np.empty(len(raw), dtype=dtype)
            for i, r in enumerate(raw):
                component = (r[0],) + tuple(float(_) for _ in r[1].split(', '))
                components[i] = component
            out['components'] = components
            return out

        elif name == 'primbeam':
            hdu =  fits.open(self.primbeam)
            header = hdu[0].header
            # Gaussian beam
            if header['format version'] == '1.0':
                fwhm0_deg = header['fwhm']
                return fwhm0_deg
            # Fitted beam
            elif header['format version'] =='2.0':
                if (self.nu < 170 and self.nu > 130):
                    omega = hdu[1].header['omega']
                    par = hdu[1].data
                else:
                    omega = hdu[2].header['omega']
                    par = hdu[2].data
                return par, omega
            # Multi frequency beam
            else:   
                 parth = hdu[1].data
                 parfr = hdu[2].data
                 parbeam = hdu[3].data
                 alpha = hdu[4].data
                 xspl = hdu[5].data
                 return parth, parfr, parbeam, alpha, xspl
                
            raise ValueError('Invalid primary beam calibration version')

        elif name == 'synthbeam':

            if self.synthbeam is None:
                print("synthbeam not defined")
                return None

            if not os.path.isfile(self.synthbeam):
                print("File not found: %s" % self.synthbeam)
                return None
            
            hdu =  fits.open(self.synthbeam)
            header = hdu[0].header
            theta=hdu[0].data
            phi=hdu[1].data
            val=hdu[2].data
            freqs=hdu[3].data

            return theta, phi, val, freqs, header
        elif name == 'synthbeam_jc':
            
            hdu =  fits.open(self.synthbeam)
            header = hdu[0].header
            theta=hdu[0].data
            phi=hdu[1].data
            val=hdu[2].data
            numpeaks=hdu[3].data
            
            return theta, phi, val,numpeaks, header

        raise ValueError("Invalid calibration item: '{}'".format(name))

# def _newest(self, filename):
#        if '*' not in filename:
    #           if not os.path.exists(filename):
    #              filename = os.path.join(self.path, filename)
    #        if not os.path.exists(filename):
    #            raise ValueError("No calibration file '{}'.".format(filename))
    #        return os.path.abspath(filename)

##        filenames = glob(filename)
#       if len(filenames) == 0:
#            filename = os.path.join(self.path, filename)
#            filenames = glob(filename)
#            if len(filenames) == 0:
#                raise ValueError("No calibration files '{}'.".format(filename))
#       regex = re.compile(filename.replace('*', '(.*)'))
    #        version = sorted(regex.search(f).group(1) for f in filenames)[-1]
#    return os.path.abspath(filename.replace('*', version))
