from preset.preset import PresetSims
from pyoperators import *

import fgb.mixing_matrix as mm
import fgb.component_model as c

from acquisition.systematics import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.foldertools import *


def _norm2(x, comm):
    x = x.ravel()
    n = np.array(np.dot(x, x))
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, n)
    return n
def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))
    
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d



class Chi2Parametric:
    
    def __init__(self, sims, d, betamap, seenpix_wrap=None):
        
        self.sims = sims
        self.d = d
        
        
        
        self.betamap = betamap
        
        if np.ndim(self.d) == 3:
            self.nc, self.nf, self.nsnd = self.d.shape
            self.constant = True
        else:
            
            if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
                pass
            else:
                self.nf = self.d.shape[1]
                self.d150 = self.d[:, :int(self.nf/2)].copy()
                self.d220 = self.d[:, int(self.nf/2):int(self.nf)].copy()
                _sh = self.d150.shape
                _rsh = ReshapeOperator(self.d150.shape, (_sh[0]*_sh[1], _sh[2], _sh[3]))
                self.d150 = _rsh(self.d150)
                self.d220 = _rsh(self.d220)
                self.dcmb150 = np.sum(self.d150[:, 0, :], axis=0).copy()
                self.dfg150 = self.d150[:, 1, :].copy()
                self.dcmb220 = np.sum(self.d220[:, 0, :], axis=0).copy()
                self.dfg220 = self.d220[:, 1, :].copy()
                self.npixnf, self.nc, self.nsnd = self.d150.shape
                
            index_num = hp.ud_grade(self.sims.seenpix_qubic, self.sims.params['Foregrounds']['nside_fit'])    #
            index = np.where(index_num == True)[0]
            self._index = index
            self.seenpix_wrap = seenpix_wrap
            self.constant = False
    def _get_mixingmatrix(self, x):
        mixingmatrix = mm.MixingMatrix(*self.sims.comps_out)
        if self.constant:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, *x)
        else:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, x)
    def __call__(self, x):
        if self.constant:
            A = self._get_mixingmatrix(x)
            self.betamap = x.copy()

            if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    ysim += A[:, ic] @ self.d[ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                for ic in range(self.nc):
                    ysim[:int(self.nsnd)] += A[:int(self.nf/2), ic] @ self.d[ic, :int(self.nf/2)]
                    ysim[int(self.nsnd):int(self.nsnd*2)] += A[int(self.nf/2):int(self.nf), ic] @ self.d[ic, int(self.nf/2):int(self.nf)]
        else:
            if self.seenpix_wrap is None:
                self.betamap[self._index, 0] = x.copy()
            else:
                self.betamap[self.seenpix_wrap, 0] = x.copy()
                
            
            #print(A.shape)
            #print(self.dcmb.shape)
            #print(self.dfg.shape)
            #print(self.sims.TOD_Q.shape)
            #stop
            if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    for ip, p in enumerate(self._index):
                        ysim += A[ip, :, ic] @ self.d[ip, :, ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                Atot = self._get_mixingmatrix(self.betamap[self._index])
                A150 = Atot[:, 0, :int(self.nf/2), 1].ravel()
                A220 = Atot[:, 0, int(self.nf/2):int(self.nf), 1].ravel()
                
                ysim[:int(self.nsnd)] = (A150 @ self.dfg150) + self.dcmb150
                ysim[int(self.nsnd):int(self.nsnd*2)] = (A220 @ self.dfg220) + self.dcmb220
                #stop
                #ysim[:int(self.nsnd)] = A150 @ 
                #for ic in range(self.nc):
                #    for ip, p in enumerate(self._index):
                #        ysim[:int(self.nsnd)] += A[ip, :int(self.nf/2), ic] @ self.d[ip, :int(self.nf/2), ic]
                #        ysim[int(self.nsnd):int(self.nsnd*2)] += A[ip, int(self.nf/2):int(self.nf), ic] @ self.d[ip, int(self.nf/2):int(self.nf), ic]

        _r = ysim - self.sims.TOD_Q
        H_planck = self.sims.joint_out.get_operator(self.betamap, 
                                                    gain=self.sims.g_iter, 
                                                    fwhm=self.sims.fwhm_recon, 
                                                    nu_co=self.sims.nu_co).operands[1]
        tod_pl_s = H_planck(self.sims.components_iter)
        
        _r_pl = self.sims.TOD_E - tod_pl_s
        #_r = np.r_[_r, _r_pl]
        #print(x)
        LLH = _dot(_r.T, self.sims.invN_beta.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN_beta.operands[1](_r_pl)
        #LLH = _r.T @ self.sims.invN.operands[0](_r)
        
        #return _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN.operands[1](_r_pl)
        return LLH


'''
class Chi2ConstantParametric:
    
    def __init__(self, sims):
        
        self.sims = sims
        self.nc = len(self.sims.comps)
        self.nsnd = self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples
        self.nsub = self.sims.joint.qubic.Nsub
        self.mixingmatrix = mm.MixingMatrix(*self.sims.comps)
        #print(self.sims.comps)
        
    def _qu(self, x, tod_comp, components, nus):
        
        A = self.mixingmatrix.eval(nus, *x)

        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(2*self.nsnd)
            
            for i in range(self.nc):
                ysim[:self.nsnd] += A[:self.nsub, i] @ tod_comp[i, :self.nsub]
                ysim[self.nsnd:self.nsnd*2] += A[self.nsub:self.nsub*2, i] @ tod_comp[i, self.nsub:self.nsub*2]
        
        elif self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            ysim = np.zeros(self.nsnd)
            
            for i in range(self.nc):
                ysim[:self.nsnd] += A[:self.nsub*2, i] @ tod_comp[i, :self.nsub*2]
        _r = self.sims.TOD_Q - ysim 
        
        H_planck = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co).operands[1]
        
        tod_pl_s = H_planck(components) 
        _r_pl = self.sims.TOD_E - tod_pl_s
        _r = np.r_[_r, _r_pl]
        #self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        #self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.chi2 = _dot(_r.T, self.sims.invN(_r), self.sims.comm)# + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.sims.comm.Barrier()
        return self.chi2
'''
class Chi2ConstantBlindJC:
    
    def __init__(self, sims):
        
        self.sims = sims
        self.nc = len(self.sims.comps_out)
        self.nf = self.sims.joint_out.qubic.Nsub
        self.nsnd = self.sims.joint_out.qubic.ndets*self.sims.joint_out.qubic.nsamples
        self.nsub = self.sims.joint_out.qubic.Nsub
    def _reshape_A(self, x):
        nf, nc = x.shape
        x_reshape = np.array([])
        for i in range(nc):
            x_reshape = np.append(x_reshape, x[:, i].ravel())
        return x_reshape
    def _reshape_A_transpose(self, x, nf):
        
        #print(x, len(x))
        nc = 1#int(len(x) / nf)
        fsub = int(nf / len(x))
        x_reshape = np.ones((nf, nc))
        #print('fsub ', fsub)
        #print('x ', x)
        if fsub == 1:
            for i in range(nc):
                x_reshape[:, i] = x[i*nf:(i+1)*nf]
        else:
            for i in range(nc):
                for j in range(len(x)):
                    #print(j*fsub, (j+1)*fsub)
                    x_reshape[j*fsub:(j+1)*fsub, i] = np.array([x[j]]*fsub)
        #print('x_rec ', x_reshape)
        #stop
        return x_reshape
    def _qu(self, x, tod_comp, A, icomp):
        #print(x, x.shape)
        x = self._reshape_A_transpose(x, 2*self.nsub)
        #print(x, x.shape)
        #stop
        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(2*self.nsnd)
            ysim[:self.nsnd] += np.sum(tod_comp[0, :self.nsub], axis=0)
            ysim[self.nsnd:self.nsnd*2] += np.sum(tod_comp[0, self.nsub:self.nsub*2], axis=0)

            for i in range(self.nc-1):
                #print(i, icomp)
                if i+1 == icomp:
                    ysim[:self.nsnd] += x[:self.nsub, 0] @ tod_comp[i+1, :self.nsub]
                    ysim[self.nsnd:self.nsnd*2] += x[self.nsub:self.nsub*2, 0] @ tod_comp[i+1, self.nsub:self.nsub*2]
                else:
                    ysim[:self.nsnd] += A[:self.nsub, i+1] @ tod_comp[i+1, :self.nsub]
                    ysim[self.nsnd:self.nsnd*2] += A[self.nsub:self.nsub*2, i+1] @ tod_comp[i+1, self.nsub:self.nsub*2]
        elif self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            ysim = np.zeros(self.nsnd)
            ysim += np.sum(tod_comp[0, :self.nsub*2], axis=0)
            for i in range(self.nc-1):
                if i+1 == icomp:
                    ysim[:self.nsnd] += x[:self.nsub*2, 0] @ tod_comp[i+1, :self.nsub*2]
                else:
                    ysim[:self.nsnd] += A[:self.nsub*2, i+1] @ tod_comp[i+1, :self.nsub*2]
        
        _r = ysim - self.sims.TOD_Q
        self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        
        return self.chi2
