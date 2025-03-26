import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

#########################

#### QUBIC IMPORT
from qubic.lib.Calibration import Qfiber

#########################

conv_reso_fwhm = 2.35482

#########################
# import matplotlib.style as style
# style.use("/Users/huchet/Documents/phd_code/matplotlib_styles/ah_basic_style.mplstyle")
# plt.rc('text', usetex=False)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{bm}")
# plt.style.use('default')

#########################


def display_one(mapsb, anatype='', sub=(1,1,1), nlo=3, nhi=3, reso=12, rot=[0,50]):
    unseen = (mapsb == hp.UNSEEN)
    mm, ss = Qfiber.meancut(mapsb[~unseen], 3)
    hp.gnomview(mapsb, rot=rot, reso=reso, sub=sub, title=anatype+'\n Both scans $\sigma$ = {0:5.3g}'.format(ss), min=-nlo*ss, max=nhi*ss)


def do_display_all(mapsb, mapsb_pos, mapsb_neg, mapav, mapdiff, mapdiff2, rot=[0,50], anatype='', reso=12, myrange=None, TESNum = None):
    unseen = (mapsb == hp.UNSEEN) | (mapsb_pos == hp.UNSEEN) | (mapsb_neg == hp.UNSEEN)
    mm, ss = Qfiber.meancut(mapsb[~unseen], 3)
    
    if myrange is None:
        mini = -3*ss
        maxi = 3*ss
    else:
        mini = myrange[0]
        maxi = myrange[1]
        
    if TESNum != None:
        anatype += '\n TES# {}'.format(TESNum)

    plt.figure()
    hp.gnomview(mapsb, rot=rot, reso=reso, sub=(2,3,1), title=anatype+'\n Both scans $\sigma$ = {0:5.4g}'.format(ss), min=mini, max=maxi)
    mmp, ssp = Qfiber.meancut(mapsb_pos[~unseen], 3)
    hp.gnomview(mapsb_pos, rot=rot, reso=reso, sub=(2,3,2), title=anatype+'\n Pos scans $\sigma$ = {0:5.4g}'.format(ssp), min=mini, max=maxi)
    mmn, ssn = Qfiber.meancut(mapsb_neg[~unseen], 3)
    hp.gnomview(mapsb_neg, rot=rot, reso=reso, sub=(2,3,3), title=anatype+'\n Neg scans $\sigma$ = {0:5.4g}'.format(ssn), min=mini, max=maxi)
    mma, ssa = Qfiber.meancut(mapav[~unseen], 3)
    hp.gnomview(mapav, rot=rot, reso=reso, sub=(2,3,4), title=anatype+'\n Av of Both scans $\sigma$ = {0:5.4g}'.format(ssa), min=mini, max=maxi)
    mmd, ssd = Qfiber.meancut(mapdiff[~unseen], 3)
    hp.gnomview(mapdiff, rot=rot, reso=reso, sub=(2,3,5), title=anatype+'\n Diff of both scans $\sigma$ = {0:5.4g}'.format(ssd), min=mini/ss*ssd, max=maxi/ss*ssd)
    mmd2, ssd2 = Qfiber.meancut(mapdiff2[~unseen], 3)
    hp.gnomview(mapdiff2, rot=rot, reso=reso, sub=(2,3,6), title=anatype+'\n Both - Av $\sigma$ = {0:5.4g}'.format(ssd2), min=mini/ss**ssd, max=maxi/ss*ssd)
    

def display_all(mapsb, mapsb_pos, mapsb_neg, anatype='', rot=[0,50], highcontrast=False, reso=12, myrange=None, TESNum=None):
    unseen = (mapsb == hp.UNSEEN) | (mapsb_pos == hp.UNSEEN) | (mapsb_neg == hp.UNSEEN)

    ### Average of back and Forth
    mapav = (mapsb_pos + mapsb_neg)/2
    mapav[unseen] = hp.UNSEEN

    ### Difference of back and Forth
    mapdiff = (mapsb_pos - mapsb_neg)
    mapdiff[unseen] = hp.UNSEEN

    ### Difference of All and Av
    mapdiff2 = (mapav - mapsb)
    mapdiff2[unseen] = hp.UNSEEN
    
    if highcontrast:
        myrange = [-np.max(mapsb[~unseen])/10, np.max(mapsb[~unseen])*0.8]
        
    do_display_all(mapsb, mapsb_pos, mapsb_neg, mapav, mapdiff, mapdiff2, rot=rot, anatype=anatype, reso=reso, myrange=myrange, TESNum=TESNum)


def plot_fit_img(mapxy, axs, x, y, xguess, yguess, xfit, yfit, vmin, vmax, ms, origin="lower"):
    ax = axs[0]
    ax.clear()
    # im = ax.imshow(mapxy, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=vmin, vmax=vmax)
    im = ax.imshow(mapxy, origin=origin, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Degrees')
    ax.plot(-xguess, yguess,'mo', markerfacecolor="none", ms=ms, mew=3, label='Guess') # j'ai dû mettre un "-" dans la définition de x, je ne sais pas pk
    ax.plot(-xfit, yfit, 'ro', markerfacecolor="none", ms=ms, mew=1.5, label='Fit')
    ax.legend()
    cax = axs[-1]
    cax.clear()
    plt.colorbar(im, cax=cax)
    return axs


class hover_cursor():
    """A class to add info on scatter points while hovering with mouse"""

    def __init__(self, fig, ax, scatter_list, scatt_name_list, all_names):
        self.fig = fig
        self.ax = ax
        self.scatter_list = scatter_list
        self.scatt_name_list = scatt_name_list
        self.all_names = all_names

        # Create the annotaion with its style
        self.annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def choose_annot_xy(self, ind_list):
        for i, ind in enumerate(ind_list):
            try:
                pos = self.scatter_list[i].get_offsets()[ind["ind"][0]]
                break
            except:
                continue
        return pos

    def update_annot(self, ind_list):
        # pos_VI = scatter_VI.get_offsets()[ind_VI["ind"][0]]
        self.annot.xy = self.choose_annot_xy(ind_list)
        # text = "TES {}".format([names_VI[n] for n in ind_VI["ind"]])
        text = "TES "
        for i, ind in enumerate(ind_list):
            if len(ind) > 0:
                text += "{}: {}".format(self.scatt_name_list[i], [self.all_names[i][n] for n in ind["ind"]])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor("white")
        self.annot.get_bbox_patch().set_alpha(0.8)
        

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont_list = []
            ind_list = []
            for scatt in self.scatter_list:
                cont_i, ind_i = scatt.contains(event)
                cont_list.append(cont_i)
                ind_list.append(ind_i)
            cont_list = np.array(cont_list)
            if np.sum(cont_list) > 0:
                self.update_annot(ind_list)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()


class hover_cursor_alpha():
    """A class to change alpha on scatter points while hovering with mouse"""

    def __init__(self, fig, ax, scatter_list, scatt_name_list, all_names):
        self.fig = fig
        self.ax = ax
        self.scatter_list = scatter_list
        self.scatt_name_list = scatt_name_list
        self.all_names = all_names

        # Create the annotaion with its style
        self.annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def choose_annot_xy(self, ind_list):
        for i, ind in enumerate(ind_list):
            try:
                pos = self.scatter_list[i].get_offsets()[ind["ind"][0]]
                break
            except:
                continue
        return pos

    def update_annot(self, ind_list):
        # pos_VI = scatter_VI.get_offsets()[ind_VI["ind"][0]]
        self.annot.xy = self.choose_annot_xy(ind_list)
        # text = "TES {}".format([names_VI[n] for n in ind_VI["ind"]])
        text = "TES "
        for i, ind in enumerate(ind_list):
            if len(ind) > 0:
                text += "{}: {}".format(self.scatt_name_list[i], [self.all_names[i][n] for n in ind["ind"]])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor("white")
        self.annot.get_bbox_patch().set_alpha(0.8)

    def update_alpha(self, ind_list):
        list_names_allind = [self.all_names[i][n] for i in range(len(self.scatter_list)) for n in ind_list[i]["ind"]]
        for i, scatt in enumerate(self.scatter_list):
            alphas = np.ones_like(self.all_names[i]) * 0.2
            wanted = np.isin(self.all_names[i], list_names_allind)
            alphas[wanted] = 1
            scatt.set_alpha(alphas)
        

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont_list = []
            ind_list = []
            for scatt in self.scatter_list:
                cont_i, ind_i = scatt.contains(event)
                cont_list.append(cont_i)
                ind_list.append(ind_i)
            cont_list = np.array(cont_list)
            if np.sum(cont_list) > 0:
                self.update_annot(ind_list)
                self.update_alpha(ind_list)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()

            else:
                if vis: # si on a changé les plots avant
                    self.annot.set_visible(False)
                    for i, scatt in enumerate(self.scatter_list):
                        scatt.set_alpha(np.ones_like(self.all_names[i])) # afficher tous les plots avec un alpha=1 si la souris n'est pas sur un point
                    self.fig.canvas.draw_idle()


def plots_identify_scans(thk, plotrange, az, medaz_dt, c0, cpos, cneg, dead_time, el, scantype_hk):
    # Needs to be adjusted
    fig, ax = plt.subplots(1, 1)
    ax.plot(thk[1:], thk[1:] - thk[:-1], '.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('dt [s]')
    ax.set_xlim(plotrange[0], plotrange[1])
    plt.show()

    fig = plt.figure(figsize=(15, 9))

    gs = fig.add_gridspec(2, 6, hspace=0.1, wspace=0.1)
    # 2 columns for 1st row, 3 columns for 2nd row
    axs = []
    axs.append(fig.add_subplot(gs[0, :3]))
    axs.append(fig.add_subplot(gs[0, 3:]))
    axs.append(fig.add_subplot(gs[1, :2]))
    axs.append(fig.add_subplot(gs[1, 2:4]))
    axs.append(fig.add_subplot(gs[1, 4:6]))

    ax = axs[0]
    ax.set_title('Angular Velocity Vs. Azimuth - Dead time = {0:4.1f}%'.format(dead_time*100))
    ax.plot(az, medaz_dt)
    ax.plot(az[c0], medaz_dt[c0], 'ro', label='Slow speed')
    ax.plot(az[cpos], medaz_dt[cpos], '.', label='Scan +')
    ax.plot(az[cneg], medaz_dt[cneg], '.', label='Scan -')
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Ang. velocity [deg/s]')
    ax.legend(loc='upper left')

    ax = axs[1]
    ax.set_title('Angular Velocity Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
    ax.plot(thk, medaz_dt)
    ax.plot(thk[c0], medaz_dt[c0], 'ro', label='speed=0')
    ax.plot(thk[cpos], medaz_dt[cpos], '.', label='Scan +')
    ax.plot(thk[cneg], medaz_dt[cneg], '.', label='Scan -')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ang. velocity [deg/s]')
    ax.set_xlim(plotrange[0],plotrange[1])

    ax = axs[2]
    ax.set_title('Azimuth Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
    ax.plot(thk, az)
    ax.plot(thk[c0], az[c0], 'ro', label='speed=0')
    ax.plot(thk[cpos], az[cpos], '.', label='Scan +')
    ax.plot(thk[cneg], az[cneg], '.', label='Scan -')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlim(plotrange[0],plotrange[1])

    ax = axs[3]
    ax.set_title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
    ax.plot(thk, el)
    ax.plot(thk[c0], el[c0], 'ro', label='speed=0')
    ax.plot(thk[cpos], el[cpos], '.', label='Scan +')
    ax.plot(thk[cneg], el[cneg], '.', label='Scan -')
    ax.legend(loc='lower left')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Elevtion [deg]')
    ax.set_xlim(plotrange[0],plotrange[1])
    elvals = el[(thk > plotrange[0]) & (thk < plotrange[1])]
    deltael = np.max(elvals) - np.min(elvals)
    ax.set_ylim(np.min(elvals) - deltael/5, np.max(elvals) + deltael/5)

    allnums = np.unique(np.abs(scantype_hk))
    for n in allnums[allnums > 0]:
        ok = np.abs(scantype_hk) == n
        xx = np.mean(thk[ok])
        yy = np.mean(el[ok])
        if (xx > plotrange[0])  & (xx < plotrange[1]):
            plt.text(xx, yy+deltael/20, str(n))

    ax = axs[4]
    ax.set_title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
    thecol = (np.arange(len(allnums))*256/(len(allnums)-1)).astype(int)
    for i in range(len(allnums)):
        ok = np.abs(scantype_hk) == allnums[i]
        ax.plot(az[ok], el[ok], color=plt.get_cmap(plt.rcParams['image.cmap'])(thecol[i]))
    ax.set_ylim(np.min(el), np.max(el))
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Elevtion [deg]')
    img = ax.scatter(-allnums*0, -allnums*0-10,c=allnums)
    aa=plt.colorbar(img)
    aa.ax.set_ylabel('Scan number')
    #plt.tight_layout()
    # sys.exit()
    plt.show()