import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
import os
import imageio
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import qubic
from qubic import selfcal_lib as sc
import warnings
warnings.filterwarnings("ignore")
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pyoperators import *
from moviepy.editor import *
from tqdm import tqdm
import matplotlib.colors as mcolors
from matplotlib import cycler
import matplotlib as mpl
import ComponentsMapMakingTools as CMM
import os.path as op

def give_stop(rms):
    notation_scientifique = "{:.2e}".format(rms)
    return 1/np.power(10, int(notation_scientifique[-1]))
def panel(datain, dataout, beta_d, convergence, nc, type='mollview', center=None, reso=None, figsize=(14, 10)):

    fig, ax = plt.subplots(nrows=nc+1, ncols=4, figsize=figsize)

    for i in range(nc):
        print(i)
        plt.axes(ax[i, 0])
        if type == 'mollview':
            hp.mollview(datain, cmap='jet', hold=True)
        else:
            hp.gnomview(datain, rot=center, reso=reso, cmap='jet', hold=True)

        plt.axes(ax[i, 1])
        if type == 'mollview':
            hp.mollview(dataout, cmap='jet', hold=True)
        else:
            hp.gnomview(dataout, rot=center, reso=reso, cmap='jet', hold=True)

        plt.axes(ax[i, 2])
        res = datain - dataout
        if type == 'mollview':
            hp.mollview(res, cmap='jet', hold=True)
        else:
            hp.gnomview(res, rot=center, reso=reso, cmap='jet', hold=True)


    plt.subplot(nc+1, 4, 4, visible=False)

    plt.subplot(nc+1, 4, 8)
    
    x = np.arange(1, len(beta_d)+1, 1)
    plt.plot(x, beta_d, '-k')

    visible=True
    if convergence is None:
        visible=False
    plt.subplot(nc+1, 4, (9,12), visible=visible)
    if visible == True:
        x = np.arange(1, len(convergence)+1, 1)
        plt.plot(x, convergence, '-k')
        plt.yscale('log')

def save_healpix_gif(cards, filename, center=None, reso=None, fps=10, min=None, max=None):
    # cards : tableau de N cartes healpix
    # filename : nom de fichier pour le GIF
    # fps : frames per second (défaut : 5)

    # créer une liste de toutes les images (cartes) à inclure dans le GIF
    images = []
    for i, card in enumerate(cards):
        # convertir la carte HEALPix en une image Matplotlib
        fig = plt.figure(figsize=(6, 5))
        hp.gnomview(card, rot=center, reso=reso, cmap='jet', fig=fig, title=f"Iteration {i}", min=min, max=max)
        plt.close(fig)
        # ajouter l'image à la liste des images
        images.append(fig_to_image(fig))

    # écrire la liste des images en tant que GIF animé
    imageio.mimsave(filename, images, fps=fps)
def save_convergence_gif(lines, filename, fps=10, truth=None, xlabel=None, ylabel=None, fontsize=10):
    # cards : tableau de N cartes healpix
    # filename : nom de fichier pour le GIF
    # fps : frames per second (défaut : 5)

    # créer une liste de toutes les images (cartes) à inclure dans le GIF
    images = []
    allines = []
    for i, line in enumerate(lines):
        allines.append(line)
        # convertir la carte HEALPix en une image Matplotlib
        fig = plt.figure(figsize=(6, 5))
        plt.plot(allines, '-r')
        plt.xlim(0, len(lines)+10)
        plt.ylim(np.min(lines)-0.001, np.max(lines)+0.001)
        if truth is not None:
            plt.axhline(truth, color='black', ls = '--')
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.close(fig)
        # ajouter l'image à la liste des images
        images.append(fig_to_image(fig))

    # écrire la liste des images en tant que GIF animé
    imageio.mimsave(filename, images, fps=fps)
def fig_to_image(fig):
    # convertir une figure Matplotlib en une image
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image
def create_folder(folder_name):
    current_directory = os.getcwd()
    path = os.path.join(current_directory, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Le dossier '{folder_name}' a été créé avec succès dans {current_directory}")
    else:
        print(f"Le dossier '{folder_name}' existe déjà dans {current_directory}")
def plot_hist(data, inputs=None, color='red', xlabel='', ylabel='', title='', bins=10, truth=None, save='test.png'):

    plt.figure(figsize=(12, 8))
    if inputs is not None:
        plt.hist(inputs, histtype='step', bins=bins, color='black')
    plt.hist(data, bins=bins, color=color)
    
    if truth is not None:
        plt.axvline(truth, color='black', ls='--')
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(fullpath+f'{save}')
    plt.close()
def plot_FP(data, s=1, colorbar='jet', min=None, max=None, title=None, save='test.png', **kwargs):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
    fontsize=18
    dictfilename = 'dicts/pipeline_demo.dict'
    
    # Read dictionary chosen
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    d['nf_recon'] = 1
    d['nf_sub'] = 1
    d['nside'] = 256
    d['RA_center'] = 0
    d['DEC_center'] = -57
    #center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
    d['effective_duration'] = 3
    d['npointings'] = 100
    d['filter_nu'] = int(220*1e9)
    d['photon_noise'] = False
    d['config'] = 'FI'
    d['MultiBand'] = True
    
    q = qubic.QubicInstrument(d)
    
    allTES=np.arange(1, 129, 1)
    good_tes=np.delete(allTES, np.array([4,36,68,100])-1, axis=0)
    xtes, ytes = np.zeros(992), np.zeros(992)
    k=0
    for j in [1, 2, 3, 4, 5, 6, 7, 8]:
        for ites, tes in enumerate(good_tes):
            xtes[k], ytes[k], FP_index, index_q= sc.TES_Instru2coord(TES=tes, ASIC=j, q=q, frame='ONAFP', verbose=False)
            k+=1
    
    img = ax.scatter(xtes, ytes, c=data, marker='s', s=s, **kwargs)
    img.set_clim(min, max)
    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes('right', size='5%', pad=0.05)
        norm = mpl.colors.Normalize(vmin=5, vmax=10)
        clb = fig.colorbar(img, cmap=colorbar, cax=cax, norm=norm)
        clb.ax.set_title('')
    
    ax.set_xlabel(f'X [m]', fontsize=fontsize)
    ax.set_ylabel(f'Y [m]', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.axis('square')
    fullpath = '/home/regnier/work/regnier/component_map_making/'
    if save is not None:
        plt.savefig(fullpath+f'{save}')
    #ax.set_title(title, fontsize=fontsize)
    plt.close()
def plot_maps(datain, dataout, center, type='mollview', reso=15, istk=0, min=None, max=None):

    if type == 'gnomview':
        hp.gnomview(datain, cmap='jet', rot=center, reso=reso, min=min, max=max, sub=(1, 3, 1))
        hp.gnomview(dataout, cmap='jet', rot=center, reso=reso, min=min, max=max, sub=(1, 3, 2))
        res=dataout-datain
        res[index_not_seen] = hp.UNSEEN
        hp.gnomview(res, cmap='jet', rot=center, reso=reso, min=None, max=None, sub=(1, 3, 3))
    elif type == 'mollview':
        hp.mollview(datain, cmap='jet', min=min, max=max, sub=(1, 3, 1))
        hp.mollview(dataout, cmap='jet', min=min, max=max, sub=(1, 3, 2))
        res=dataout-datain
        res[index_not_seen] = hp.UNSEEN
        hp.mollview(res, cmap='jet', min=min, max=max, sub=(1, 3, 3))  
def convergence_of_X(data, style, label=None, truth=None, log=False):
    x = np.arange(1, len(data)+1, 1)
    plt.plot(x, data, style, label=label)
    if log:
        plt.yscale('log')
    
    if truth is not None:
        plt.axhline(truth, ls='--', color='black')
def scatter_plot(xdata, ydata, s=15, c='blue', xlabel='', ylabel='', xequaly=False, xbounds=None, ybounds=None, save='test.png'):

    minx, maxx = xbounds
    miny, maxy = ybounds
    plt.figure()
    plt.scatter(xdata, ydata, s=s, c=c)
    xsimu = np.linspace(np.min(xdata), np.max(xdata), 100)
    if xequaly:
        plt.plot(xsimu, xsimu, color='black', ls=':')

    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.savefig(fullpath+f'{save}')
    plt.close()
def get_gif(name, fps=100):
    

    images = []
    path = "/Users/mregnier/Desktop/images/"
    filenames = [f for f in os.listdir(path) if f.endswith('.png')]
    filenames.sort()
    x = np.arange(1, len(filenames), 1)

    for i in x:
        print(i)
        images.append(imageio.imread(os.path.join(path, f'Res_FP_gain_150_{i}.png')))
        
    imageio.mimsave(f'{name}.gif', images, fps=fps)
def plot_one_map(map, rot, reso, min, max, title='I', save=None):
    hp.gnomview(map, cmap='jet', rot=rot, reso=reso, min=min, max=max, sub=(1, 1, 1), title=title)
def gif_to_mp4(gif_path):
    video = VideoFileClip(gif_path)
    mp4_path = gif_path[:-3] + "mp4"  # on remplace le suffixe du fichier gif par mp4
    video.write_videofile(mp4_path, fps=24)  # conversion en mp4 avec un framerate de 24


class Analysis:

    def __init__(self, path, nite, nside, nc, nside_fit, thr, convolution=False, path_to_save=None):

        self.nite = nite
        self.nstk = 3
        self.nc = nc
        self.nside = nside
        self.nside_fit = nside_fit
        self.convolution = convolution
        self.convergence = np.zeros(self.nite)
        self.path_to_save = path_to_save
        nell=2*self.nside - 2
        self.ell = np.arange(2, 2*self.nside, 1)

        s = CMM.Spectra(40, 2*self.nside, 35, CMB_CL_FILE=op.join('/home/regnier/work/regnier/mypackages/Cls_Planck2018_%s.fits'))
        self.dl_theo = s.dl_theo
        self.ell_theo = s.ell_theo
        

        current_path = os.getcwd() + '/'
        if not os.path.exists(current_path + self.path_to_save):
            os.makedirs(current_path + self.path_to_save)

        if self.nside_fit != 0:
            self.maps = np.zeros((self.nite, self.nstk, 12*self.nside**2, self.nc))
            self.beta = np.zeros((self.nite, 12*self.nside_fit**2, 2))
            self.gain = np.zeros((self.nite, 2, 992))
            self.cl = np.zeros((self.nite, self.nc, nell))
        else:
            self.maps = np.zeros((self.nite, self.nc, 12*self.nside**2, self.nstk))
            self.beta = np.zeros((self.nite, 2))
            self.gain = np.zeros((self.nite, 2, 992))
            self.cl = np.zeros((self.nite, self.nc, nell))
        
        for i in range(self.nite):
            
            pkl_file = open(path+f'Iter{i}_maps_beta_gain_rms_maps.pkl', 'rb')
            dataset = pickle.load(pkl_file)
            self.pixok = dataset['coverage']/dataset['coverage'].max() > thr
            if i == 0:
                self.maps[i] = dataset['initial']
                self.true_maps = dataset['maps']
            else:
                self.maps[i] = dataset['maps']
            
            self.beta[i] = dataset['beta']
            self.cl[i, 0] = dataset['spectra_cmb']
            self.cl[i, 1] = dataset['spectra_dust']
            if self.nside_fit == 0:
                print(i, dataset['beta'])
            self.gain[i] = dataset['gain']
            self.convergence[i] = dataset['convergence']
            self.maps[i, :, ~self.pixok, :] = hp.UNSEEN
        fwhm = np.min(dataset['allfwhm'])
        if self.convolution:
            print(fwhm)
            self.C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        else:
            self.C = IdentityOperator()

        if self.nside_fit != 0:
            for i in range(self.nc):
                self.true_maps[i] = self.C(self.true_maps[i])
        else:
            for i in range(self.nc):
                self.true_maps[i, :, :] = self.C(self.true_maps[i, :, :])

        self.true_maps[:, ~self.pixok, :] = hp.UNSEEN
        colors_green = ['#000000', '#00FF00', '#FFFFFF']  #'#00008B'
        colors_blue = ['#000000', '#00008B', '#FFFFFF']  #''
        colors_red = ['#000000', '#FF0000', '#FFFFFF']  #''
        self.cmap_green = mcolors.LinearSegmentedColormap.from_list("", colors_green)
        self.cmap_blue = mcolors.LinearSegmentedColormap.from_list("", colors_blue)
        self.cmap_red = mcolors.LinearSegmentedColormap.from_list("", colors_red)
        

    def plot_rms_maps(self, i, comp_name, figsize=(8, 6), ylog=True):
        
        print('\n    ***** RMS *****    \n')
        cmap = plt.cm.coolwarm
        mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, self.nc)))

        plt.figure(figsize=figsize)
        if self.nside_fit == 0:
            for icomp in range(self.nc):
                
                print(f'Doing for {comp_name[icomp]}')
                
                resI = self.maps[:i, icomp, self.pixok, 0] - self.true_maps[icomp, self.pixok, 0]
                resQ = self.maps[:i, icomp, self.pixok, 1] - self.true_maps[icomp, self.pixok, 1]
                resU = self.maps[:i, icomp, self.pixok, 2] - self.true_maps[icomp, self.pixok, 2]
                
                plt.subplot(2, 1, 1)

                if icomp == 0:
                    plt.plot([0], np.std(resI, axis=1)[0], '-k', label='I')
                    plt.plot([0], np.std(resI, axis=1)[0], '--k', label='Q')
                    plt.plot([0], np.std(resI, axis=1)[0], ':k', label='U')
                    plt.plot([0], np.std(resI, axis=1)[0], '-', label=comp_name[icomp])
                    
                else:
                    plt.plot([0], np.std(resI, axis=1)[0], '-', label=comp_name[icomp])
                
                plt.plot(np.std(resI, axis=1), '-', color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][icomp]['color'])
                plt.legend(frameon=False, fontsize=10, loc=1)
                
                plt.ylabel(r'RMS [$\mu K^2$]', fontsize=12)

                if ylog:
                    plt.yscale('log')

                plt.subplot(2, 1, 2)
                plt.plot(np.std(resQ, axis=1), '--', color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][icomp]['color'])
                plt.plot(np.std(resU, axis=1), ':', color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][icomp]['color'])

                plt.ylabel(r'RMS $[\mu K^2]$', fontsize=12)
                plt.xlabel('Iterations', fontsize=12)

                if ylog:
                    plt.yscale('log')

                   
        
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + 'RMS_maps.png')
        plt.close()
    def plot_FP_gain(self, i, iFP, figsize=(16, 6), s=40, vmin=0.8, vmax=1.2):

        print('\n    ***** FP gain *****    \n')

        dictfilename = 'dicts/pipeline_demo.dict'
    
        # Read dictionary chosen
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
    
        q = qubic.QubicInstrument(d)
    
        allTES=np.arange(1, 129, 1)
        good_tes=np.delete(allTES, np.array([4,36,68,100])-1, axis=0)
        xtes, ytes = np.zeros(992), np.zeros(992)
        k=0
        for j in [1, 2, 3, 4, 5, 6, 7, 8]:
            for ites, tes in enumerate(good_tes):
                xtes[k], ytes[k], FP_index, index_q = sc.TES_Instru2coord(TES=tes, ASIC=j, q=q, frame='ONAFP', verbose=False)
                k+=1

        ####### From here

        plt.figure(figsize=figsize)

        colormap = plt.cm.get_cmap('bwr')
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)

        plt.subplot(1, 2, 1)
        img = plt.scatter(xtes, ytes, c=self.gain[0, iFP], marker='s', s=s, cmap=colormap)
        plt.colorbar(sm)

        plt.xlabel('X [m]', fontsize=12)
        plt.ylabel('Y [m]', fontsize=12)
        plt.title('Input', fontsize=12)
        
        plt.subplot(1, 2, 2)
        img = plt.scatter(xtes, ytes, c=self.gain[-1, iFP], marker='s', s=s, cmap=colormap)
        plt.colorbar(sm)

        plt.xlabel('X [m]', fontsize=12)
        plt.ylabel('Y [m]', fontsize=12)
        plt.title('Output', fontsize=12)

        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'FP{iFP}_gain.png')

        plt.close()
    def plot_hist_residuals_gain(self, i, iFP, c, bins=20, figsize=(6, 6)):
        
        print('\n    ***** Histo gain *****    \n')

        res = self.gain[i, iFP] - self.gain[0, iFP]

        plt.figure(figsize=figsize)

        plt.hist(res, bins=bins, color=c)

        plt.axvline(0, color='black')

        plt.xlabel(r'$g_{input} - g_{recon}$', fontsize=12)

        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'histo_residuals_FP{iFP}_gain.png')

        plt.close()
    def plot_maps(self, i, center, reso, istk, comp_name, figsize=(8, 6), min=-8, max=8, rmin=-8, rmax=8):


        stokes = ['I', 'Q', 'U']

        print('\n    ***** Maps - {} *****    \n'.format(i))


        fig = plt.figure(figsize=figsize)

        k=0
        for icomp in range(self.nc):

            sig_in = np.std(self.true_maps[icomp, self.pixok, istk])

            if icomp == 0:
                nsig_i = 4
                nsig_s = 4
                cmap = 'jet'
            else:
                nsig_i = 0
                nsig_s = 4
                if icomp == 1:
                    cmap = self.cmap_red
                elif icomp == 2:
                    cmap = self.cmap_green
                elif icomp == 3:
                    cmap = self.cmap_blue

            # Input
            hp.gnomview(self.true_maps[icomp, :, istk], rot=center, reso=reso, sub=(self.nc, 3, (icomp*3)+(k+1)),
                                min=-nsig_i*sig_in, max=nsig_s*sig_in, cmap=cmap, title=f'{comp_name[icomp]} - Input', notext=True)

            # Reconstructed
            hp.gnomview(self.maps[i, icomp, :, istk], rot=center, reso=reso, sub=(self.nc, 3, (icomp*3)+(k+2)), 
                                min=-nsig_i*sig_in, max=nsig_s*sig_in, cmap=cmap, title=f'{comp_name[icomp]} - Output', notext=True)

            # Residual
            myres = self.true_maps[icomp, :, istk]-self.maps[i, icomp, :, istk]
            myres[~self.pixok] = hp.UNSEEN

            sig_in = np.std(myres[self.pixok])
            nsig = 2

            hp.gnomview(myres, rot=center, reso=reso, sub=(self.nc, 3, (icomp*3)+(k+3)), 
                                min=-nsig*sig_in, max=nsig*sig_in, cmap='jet', title=f'{comp_name[icomp]} - Residual', notext=True)

    
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'maps_{stokes[istk]}.png')

        plt.close()
        return fig
    def gif_maps(self, fps, istk, center, reso, comp_name, figsize, min=-8, max=8, rmin=-8, rmax=8):
        images = []
        stokes = ['I', 'Q', 'U']
        for i in range(self.nite):
            fig = self.plot_maps(i, center=center, reso=reso, istk=istk, comp_name=comp_name, figsize=figsize, min=min, max=max, rmin=rmin, rmax=rmax)

            plt.close(fig)
            images.append(fig_to_image(fig))

        # écrire la liste des images en tant que GIF animé
        imageio.mimsave(self.path_to_save + f'maps_{stokes[istk]}.gif', images, fps=fps)
    def plot_spectra(self, i, comp_name, log=True, figsize=(8, 8), title=True):
        
        cmap = plt.cm.coolwarm
        mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, self.nc)))


        fig = plt.figure(figsize=figsize)

        plt.plot(self.ell_theo, self.dl_theo, '--k')
        for icomp in range(2):
            plt.plot(self.ell[10:], self.cl[i, icomp, 10:], '-', color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][icomp]['color'], label=comp_name[icomp])
        
        if log:
            plt.yscale('log')

        plt.ylabel(r'$\mathcal{D}_{\ell}$')
        plt.xlabel(r'$\ell$')
        plt.ylim(5e-6, 2e1)
        alliteration = np.arange(0, self.nite, 1)
        if title:
            plt.title(f'Iteration {alliteration[i]}')
        plt.legend(frameon=False, fontsize=12)
        plt.xlim(20, 400)
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'cl.png')
        plt.close()
        
        return fig
    def make_gif_spectra(self, fps, comp_name, log=True, figsize=(8, 6)):

        images = []
        for ite in range(1, self.nite):
            print(f'Doing GIF for iteration {ite}')
            fig = self.plot_spectra(ite, comp_name=comp_name, log=log, figsize=figsize)

            plt.close(fig)
            images.append(fig_to_image(fig))

        # écrire la liste des images en tant que GIF animé
        imageio.mimsave(self.path_to_save + f'cl.gif', images, fps=fps)
    def plot_spectra_few_iterations(self, iterations, comp_name, log=True, figsize=(8, 6), title=False):
        cmap = plt.cm.jet#coolwarm
        mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(iterations))))


        fig = plt.figure(figsize=figsize)

        plt.plot([0], [0], '-k', label='CMB')
        plt.plot([0], [0], ':k', label='DUST')

        for ii, ite in enumerate(iterations):
            plt.plot([0], [0], '-', color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][ii]['color'], label=f'Iteration {ite}', alpha=0.6)

        plt.plot(self.ell_theo, self.dl_theo, '--k', label=r'Theoretical CMB spectra - $r = 0$ '+r'| $A_{lens} = 1$')
        for ii, ite in enumerate(iterations):
            for icomp in range(2):
                if icomp == 0:
                    style = '-'
                else:
                    style = ':'

                plt.plot(self.ell[10:], self.cl[ite, icomp, 10:], style, color=mpl.rcParams['axes.prop_cycle'].__dict__['_left'][ii]['color'], alpha=0.6)
        
        if log:
            plt.yscale('log')

        plt.ylabel(r'$\mathcal{D}_{\ell}$')
        plt.xlabel(r'$\ell$')
        plt.ylim(5e-6, 2e1)
        alliteration = np.arange(0, self.nite, 1)
        if title:
            plt.title(f'Iteration {alliteration[i]}')
        plt.legend(frameon=False, fontsize=10)
        plt.xlim(20, 400)
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'cl_few_iterations.png')
        plt.close()
        
        return fig
    def plot_maps_without_input(self, i, center, reso, istk, comp_name, figsize=(8, 6), min=-8, max=8, rmin=-8, rmax=8):
        stokes = ['I', 'Q', 'U']

        print('\n    ***** Maps - {} *****    \n'.format(i))


        fig = plt.figure(figsize=figsize)

        k=0
        for icomp in range(self.nc):

            if icomp == 0 :
                cbar = True
                title = 'Output'
                titler = 'Residual'
            else:
                cbar = True
                title = ''
                titler = ''

            sig_out = np.std(self.maps[i, icomp, self.pixok, istk])
                
        
            # Reconstructed
            hp.gnomview(self.maps[i, icomp, :, istk], rot=center, reso=reso, sub=(self.nc, 2, (icomp*2)+(k+1)),
                                min=-3*sig_out, max=3*sig_out, cmap='jet', title=title, notext=True, cbar=cbar, unit=r'$\mu K_{CMB}$')

            # Residual
            myres = self.true_maps[icomp, :, istk]-self.maps[i, icomp, :, istk]
            myres[~self.pixok] = hp.UNSEEN
            hp.gnomview(myres, rot=center, reso=reso, sub=(self.nc, 2, (icomp*2)+(k+2)), 
                                min=rmin, max=rmax, cmap='jet', title=titler, notext=True, cbar=cbar, unit=r'$\mu K_{CMB}$')

            all_comp = [r'$CMB$', r'$A_d$']
            all_pol = [r'$I$', r'$Q$', r'$U$']

            for ic in range(len(all_comp)):
                plt.annotate(all_comp[ic], xy=(0, 0), xytext=(1/len(all_comp) - 0.08, 1/(ic+1) - 0.08), 
                 xycoords='figure fraction', fontsize=14, ha="center", va="center")

                plt.annotate(all_pol[istk], xy=(0, 0), xytext=(0.08, 1/(ic+1) - 0.08), 
                xycoords='figure fraction', fontsize=14, ha="center", va="center")

        plt.tight_layout()
    
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'maps_without_inputs_{stokes[istk]}.png')

        plt.close()
        return fig
    def plot_beta(self, i, figsize=(6, 6), truth=None):

        fig = plt.figure(figsize=figsize)

        plt.plot(self.beta[:i, 0], '-k', alpha=0.5, label=r'$\beta_d$ = {:.5f}'.format(self.beta[i, 0]))

        if truth is not None:
            plt.axhline(truth, ls='--', color='red', lw=2)
        plt.legend(frameon=False, fontsize=12)
        if self.path_to_save is not None:
            plt.savefig(self.path_to_save + f'beta.png')

        plt.close()







