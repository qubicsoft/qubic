import numpy as np
from pyoperators import (
    HomothetyOperator, DiagonalOperator, IdentityOperator,
    Rotation2dOperator, Rotation3dOperator, ReshapeOperator,
    DenseBlockDiagonalOperator,
)
from pysimulators import ConvolutionTruncatedExponentialOperator, ProjectionOperator


class ForwardOps:
    """
    Class to build the forward operators of the QUBIC instrument model.
    Each method returns a pyoperators operator.
    The operators can be composed to form the full forward model.
    """

    def __init__(self, qubic_instrument, qubic_acquisition=None, qubic_scene=None):
        self.qi  = qubic_instrument
        self.acq = qubic_acquisition
        self.scn = qubic_scene


    def op_bolometer_response(self, tau=None):
        """Bolometer response as truncated-exponential convolution on TOD."""
        
        if self.acq is None:
            raise ValueError("op_bolometer_response needs qubic_acquisition.")
        if tau is None:
            tau = self.qi.detector.tau
        Ts = self.acq.sampling.period
        shapein = (len(self.qi), len(self.acq.sampling))
        if Ts == 0:
            return IdentityOperator(shapein)
        return ConvolutionTruncatedExponentialOperator(tau / Ts, shapein=shapein)

    def op_transmission(self):
        """Diagonal throughput: optics transmission and detector efficiency."""

        transmission = (
            np.prod(self.qi.optics.components['transmission'])
            * self.qi.detector.efficiency
        )
        return DiagonalOperator(transmission, broadcast='rightward')

    def op_detector_integration(self):
        """Detector solid angle times secondary beam transmission."""

        pos  = self.qi.detector.center
        area = self.qi.detector.area
        secb = self.qi.secondary_beam
        theta = np.arctan2(np.sqrt((pos[..., :2]**2).sum(-1)), pos[..., 2])
        phi   = np.arctan2(pos[..., 1], pos[..., 0])
        sr_det  = -area / pos[..., 2]**2 * np.cos(theta)**3
        sr_beam = secb.solid_angle
        gain    = secb(theta, phi)
        return DiagonalOperator(sr_det / sr_beam * gain, broadcast='rightward')

    def op_polarizer(self):
        """Polarizer grid (if there is no polarizer it is 1 for I only)."""

        if self.acq is None or self.scn is None:
            raise ValueError("op_polarizer needs acquisition and scene.")
        nd = len(self.qi)
        nt = len(self.acq.sampling)
        grid = (self.qi.detector.quadrant - 1) // 4
        if self.scn.kind == 'I':
            if self.qi.optics.polarizer:
                return HomothetyOperator(1/2)
            return DiagonalOperator(1 - grid, shapein=(nd, nt), broadcast='rightward')
        if not self.qi.optics.polarizer:
            raise NotImplementedError('Polarized input without polarizer.')
        z = np.zeros(nd)
        data = np.array([z + 0.5, 0.5 - grid, z]).T[:, None, None, :]  # (nd,1,1,3)
        return ReshapeOperator((nd, nt, 1), (nd, nt)) * \
               DenseBlockDiagonalOperator(data, shapein=(nd, nt, 3))

    def op_hwp(self):
        """Rotation matrix of the halfwave plate."""

        if self.acq is None or self.scn is None:
            raise ValueError("op_hwp needs acquisition and scene.")
        shape = (len(self.qi), len(self.acq.sampling))
        ang = -4 * self.acq.sampling.angle_hwp
        if self.scn.kind == 'I':
            return IdentityOperator(shapein=shape)
        if self.scn.kind == 'QU':
            return Rotation2dOperator(ang, degrees=True, shapein=shape + (2,))
        return Rotation3dOperator('X', ang, degrees=True, shapein=shape + (3,))

    def op_filter(self):
        """Band integration (or identity if zero bandwidth)."""
        bw = self.qi.filter.bandwidth
        return IdentityOperator() if bw == 0 else HomothetyOperator(bw)

    def op_aperture_integration(self):
        """    Integrate flux density in the telescope aperture."""

        nhorns = np.sum(self.qi.horn.open)
        return HomothetyOperator(nhorns * np.pi * self.qi.horn.radeff**2)

    def op_atmosphere(self):
        """Atmospheric transmission operator."""

        if self.acq is None:
            raise ValueError("op_atmosphere needs acquisition.")
        return self.acq.scene.atmosphere.transmission

    def op_unit_conversion(self):
        """Brightness to power conversion (Planck or linearized)."""
        if self.scn is None:
            raise ValueError("op_unit_conversion needs scene.")
        nu = self.qi.filter.nu
        return self.scn.get_unit_conversion_operator(nu)

    # ---- projection helpers  ----

    def op_projection(self):
        """
        Projection operator(s).
        - If self.acq has .get_projection_operator(): return that operator (mono).
        - If self.acq has .subacqs: return list [P_i for each subacq] (multi).
        """
        if self.acq is None:
            raise ValueError("op_projection needs acquisition.")
        if hasattr(self.acq, "get_projection_operator"):
            return self.acq.get_projection_operator()
        if hasattr(self.acq, "subacqs") and self.acq.subacqs is not None:
            return [sub.get_projection_operator() for sub in self.acq.subacqs]
        raise TypeError("Acquisition does not expose projection operators.")

    @staticmethod
    def extract_projection_operators(H_list):
        """From composite H per subband, return [P_i]."""
        Ps = []
        for H in H_list:
            P = ForwardOps._find_projection_in(H)
            if P is None:
                raise ValueError("No ProjectionOperator found in a provided H.")
            Ps.append(P)
        return Ps

    @staticmethod
    def _find_projection_in(op):
        if isinstance(op, ProjectionOperator):
            return op
        if hasattr(op, "operands"):
            for child in op.operands:
                found = ForwardOps._find_projection_in(child)
                if found is not None:
                    return found
        return None
