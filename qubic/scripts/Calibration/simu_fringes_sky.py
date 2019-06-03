from matplotlib.pyplot import *
import healpy as hp
import qubic

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)


def get_synthetic_beam_sky(d, switches, tes, plot_loc, title_plot, default_open=True):
    """
    Return a qubic instrument with closed horns and the synthetic beam
    projected on the sky for a given TES. Plot the horn matrix and the
    synthetic beam.

    Parameters
    ----------
    d : dictionary
    switches : list of int
        Index of switches between 1 and 64 that you want to close or open.
    tes : int
        TES number for which you reconstruct the synthetic beam.
    plot_loc : tuple
        Subplot number, the third index should be odd.
    title_plot : str
        Title for the figure.
    default_open : bool
        If True, all switches are open except the ones in switches.
        If False, all switches are close except the one in switches.
        True by default.

    Returns
    -------
    The qubic instrument and the synthetic beam.

    """
    q = qubic.QubicInstrument(d)
    s = qubic.QubicScene(d)

    if default_open:
        q.horn.open = True
        for i in switches:
            q.horn.open[i - 1] = False
    else:
        q.horn.open = False
        for i in switches:
            q.horn.open[i - 1] = True
    sb = q.get_synthbeam(s, idet=tes)

    subplot(plot_loc[0], plot_loc[1], plot_loc[2])
    q.horn.plot()
    axis('off')
    hp.gnomview(sb, sub=(plot_loc[0], plot_loc[1], plot_loc[2] + 1), rot=(0, 90), reso=5, xsize=350, ysize=350,
                title=title_plot, cbar=True, notext=True)
    return q, sb


# TES number
tes_number = 39

# Baseline of the switches
switch_1 = 46
switch_2 = 64
baseline = [switch_1, switch_2]

figure('Synthetic beam on the sky, all configurations')
q_full_open, sb_full_open = get_synthetic_beam_sky(d, switches=[], tes=tes_number,
                                                   plot_loc=(4, 4, 1), title_plot='$S_{tot}$',
                                                   default_open=True)

q_both_close, sb_both_close = get_synthetic_beam_sky(d, switches=baseline, tes=tes_number,
                                                     plot_loc=(4, 4, 3), title_plot='$S_{-ij}$',
                                                     default_open=True)

q1_close, sb1_close = get_synthetic_beam_sky(d, switches=[switch_1], tes=tes_number,
                                             plot_loc=(4, 4, 5), title_plot='$C_{-i}$',
                                             default_open=True)

q2_close, sb2_close = get_synthetic_beam_sky(d, switches=[switch_2], tes=tes_number,
                                             plot_loc=(4, 4, 7), title_plot='$C_{-j}$',
                                             default_open=True)

q1_open, sb1_open = get_synthetic_beam_sky(d, switches=[switch_1], tes=tes_number,
                                           plot_loc=(4, 4, 9), title_plot='$C_{i}$',
                                           default_open=False)

q2_open, sb2_open = get_synthetic_beam_sky(d, switches=[switch_2], tes=tes_number,
                                           plot_loc=(4, 4, 11), title_plot='$C_{j}$',
                                           default_open=False)

q_only2open, sb_only2open = get_synthetic_beam_sky(d, switches=baseline, tes=tes_number,
                                                   plot_loc=(4, 4, 13), title_plot='$S_{ij}$',
                                                   default_open=False)

figure('Fringes on the sky')
hp.gnomview(sb_only2open, sub=(2, 2, 1), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$S_{ij}$', cbar=True,
            notext=True)

hp.gnomview(sb_full_open + sb_both_close - sb1_close - sb2_close + sb1_open + sb2_open, sub=(2, 2, 2), rot=(0, 90),
            reso=5, xsize=350,
            ysize=350,
            title='$S_{-ij} - C_{-i} - C_{-j} + C_{i} + C_{j}+ S_{tot}$', cbar=True, notext=True)

hp.gnomview(sb_full_open + sb_both_close - sb1_close - sb2_close, sub=(2, 2, 3), rot=(0, 90), reso=5, xsize=350,
            ysize=350,
            title='$S_{-ij} - C_{-i} - C_{-j} + S_{tot}$', cbar=True, notext=True)
