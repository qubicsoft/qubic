from pylab import *
import numpy as np
import iminuit
from iminuit.cost import LeastSquares
import numba_stats
from numba_stats import norm

from qubic import fibtools as ft



class Data:
    def __init__(self, x, y, cov, model, pnames=None):
        self.x = x
        self.y = y
        self.model = model
        self.cov = cov
        if np.prod(np.shape(x)) == np.prod(np.shape(cov)):
            self.diag = True
            self.errors = cov
        else:
            self.diag = False
            self.errors = 1./np.sqrt(cov)
            self.invcov = np.linalg.inv(cov)
        self.fit = None
        self.fitinfo = None
        self.pnames = pnames
        
    def __call__(self):
        return 0

    def plot(self, nn=1000, color=None, mylabel=None, nostat=False):
        p=errorbar(self.x, self.y, yerr=self.errors, fmt='o', color=color, alpha=1)
        if self.fit is not None:
            xx = np.linspace(np.min(self.x), np.max(self.x), nn)
            plot(xx, self.model(xx, self.fit), color=p[0].get_color(), alpha=1, label=mylabel)
        if mylabel is None:
            if nostat == False:
                legend(title="\n".join(self.fit_info))
        else:
            legend()


    def fit_minuit(self, guess, fixpars = None, limits=None, scan=None, renorm=False, simplex=False, minimizer=LeastSquares):
        ok = np.isfinite(self.x) & (self.errors != 0)

        ### Prepare Minimizer
        if self.diag == True:
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)
        else:
            print('Non diagoal covariance not yet implemented: using only diagonal')
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)

        ### Instanciate the minuit object
        if simplex == False:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames)
        else:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames).simplex()
        
        ### Limits
        if limits is not None:
            mylimits = []
            for k in range(len(guess)):
                mylimits.append((None, None))
            for k in range(len(limits)):
                mylimits[limits[k][0]] = (limits[k][1], limits[k][2])
            m.limits = mylimits

        ### Fixed parameters
        if fixpars is not None:
            for k in range(len(guess)):
                m.fixed["x{}".format(k)]=False
            for k in range(len(fixpars)):
                m.fixed["x{}".format(fixpars[k])]=True

        ### If requested, perform a scan on the parameters
        if scan is not None:
            m.scan(ncall=scan)

        ### Call the minimization
        m.migrad()  

        ### accurately computes uncertainties
        m.hesse()   

        ch2 = m.fval
        ndf = len(self.x[ok]) - m.nfit
        self.fit = m.values

        self.fit_info = [
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
        ]
        for i in range(len(guess)):
            vi = m.values[i]
            ei = m.errors[i]
            self.fit_info.append(f"{m.parameters[i]} = ${vi:.3f} \\pm {ei:.3f}$")

        if renorm:
            m.errors *= 1./np.sqrt(ch2/ndf)

        return m, ch2, ndf


def gauss(x, pars):
    return pars[0]*np.exp(-0.5*(x-pars[1])**2/pars[2]**2)

def gauss_pdf(x, mu, sigma):
    return (numba_stats.norm.pdf(x, mu, sigma))


def myhist(x, unbinned=True, nsig=4, **kwargs):
    thelabel = None
    if 'alpha' not in kwargs:
        kwargs['alpha']=0.5
    if 'label' in kwargs:
        thelabel = kwargs['label']
        kwargs['label'] = ''
    else:
        thelabel=''
    if 'range' not in kwargs:
        mm, ss = ft.meancut(x, 3)
        kwargs['range'] = [mm-nsig*ss, mm+nsig*ss]
        forcerng = False
    else:
        forcerng = True
            
    if unbinned:
        c = iminuit.cost.UnbinnedNLL(x, gauss_pdf)
        mm, ss = ft.meancut(x, 3)
        m = iminuit.Minuit(c, mu=mm, sigma=ss)
        m.migrad()
        m.hesse()
        mu = m.values['mu']
        sigma = m.values['sigma']
        if forcerng is False:
            kwargs['range'] = [mu-nsig*sigma, mu+nsig*sigma]
        yy, xe, a = hist(x, **kwargs)
        xx = 0.5 * (xe[1:] + xe[:-1])
        dx = np.diff(xe)
        notzero = yy != 0    
        p=errorbar(xx[notzero], yy[notzero], yerr=np.sqrt(yy[notzero]), fmt='o', color=a[0].get_facecolor(), alpha=1)
        xm = np.linspace(np.min(xx), np.max(xx), 100)
        plot(xm, gauss_pdf(xm, *m.values) * len(x) * dx[0], color=a[0].get_facecolor(), alpha=1, label = thelabel+' {0:5.2g} $\pm$ {1:5.2g}'.format(mu, sigma))
    else:
        yy, xe, a = hist(x, **kwargs)
        xx = 0.5 * (xe[1:] + xe[:-1])
        dx = np.diff(xe)
        notzero = yy != 0
        dd = Data(xx[notzero], yy[notzero], np.sqrt(yy[notzero]), gauss)
        guess = np.array([len(xx), np.mean(x), np.std(x)])
        dd.fit_minuit(guess) 
        mu = dd.fit[1]
        sigma = dd.fit[2]
        dd.plot(color=a[0].get_facecolor(), mylabel=None, nostat=True)
        label = ' {0:5.2g} $\pm$ {1:5.2g}'.format(mu, sigma)
        a.set_label(thelabel + label)
        
    legend(fontsize=10)

