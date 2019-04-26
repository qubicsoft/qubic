from matplotlib.pyplot import *
import healpy as hp
import qubic

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Scene
s = qubic.QubicScene(d)

# TES number
tes = 39

# Baseline of the switches
switch_1 = 46
switch_2 = 64

q_full_open = qubic.QubicInstrument(d)
sb_full_open = q_full_open.get_synthbeam(s, idet=tes)

q_both_close = qubic.QubicInstrument(d)
q_both_close.horn.open[switch_1 - 1] = False
q_both_close.horn.open[switch_2 - 1] = False
sb_both_close = q_both_close.get_synthbeam(s, idet=tes)

q1_close = qubic.QubicInstrument(d)
q1_close.horn.open[switch_1 - 1] = False
sb1_close = q1_close.get_synthbeam(s, idet=tes)

q2_close = qubic.QubicInstrument(d)
q2_close.horn.open[switch_2 - 1] = False
sb2_close = q2_close.get_synthbeam(s, idet=tes)

q1_open = qubic.QubicInstrument(d)
q1_open.horn.open = False
q1_open.horn.open[switch_1 - 1] = True
sb1_open = q1_open.get_synthbeam(s, idet=tes)

q2_open = qubic.QubicInstrument(d)
q2_open.horn.open = False
q2_open.horn.open[switch_2 - 1] = True
sb2_open = q2_open.get_synthbeam(s, idet=tes)

q_only2open = qubic.QubicInstrument(d)
q_only2open.horn.open = False
q_only2open.horn.open[switch_1-1] = True
q_only2open.horn.open[switch_2-1] = True
sb_only2open = q_only2open.get_synthbeam(s, idet=tes)


figure('Synthetic beam on the sky, all configurations')
subplot(441)
q_full_open.horn.plot()
axis('off')
hp.gnomview(sb_full_open, sub=(4, 4, 2), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$S_{tot}$', cbar=True,
            notext=True)

subplot(443)
q_only2open.horn.plot()
axis('off')
hp.gnomview(sb_only2open, sub=(4, 4, 4), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$S_{ij}$', cbar=True,
            notext=True)

subplot(445)
q_both_close.horn.plot()
axis('off')
hp.gnomview(sb_both_close, sub=(4, 4, 6), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$S_{-ij}$', cbar=True,
            notext=True)

subplot(447)
q1_close.horn.plot()
axis('off')
hp.gnomview(sb1_close, sub=(4, 4, 8), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$C_{-i}$', cbar=True,
            notext=True)

subplot(449)
q2_close.horn.plot()
axis('off')
hp.gnomview(sb2_close, sub=(4, 4, 10), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$C_{-j}$', cbar=True,
            notext=True)

subplot(4,4,11)
q1_open.horn.plot()
axis('off')
hp.gnomview(sb1_open, sub=(4, 4, 12), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$C_{i}$', cbar=True,
            notext=True)

subplot(4,4,13)
q2_open.horn.plot()
axis('off')
hp.gnomview(sb2_open, sub=(4, 4, 14), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$C_{j}$', cbar=True,
            notext=True)


figure('Fringes on the sky')
hp.gnomview(sb_only2open, sub=(2, 2, 1), rot=(0, 90), reso=5, xsize=350, ysize=350, title='$S_{ij}$', cbar=True,
            notext=True)

hp.gnomview(sb_full_open + sb_both_close - sb1_close - sb2_close + sb1_open + sb2_open, sub=(2, 2, 2), rot=(0, 90), reso=5, xsize=350,
            ysize=350,
            title='$S_{-ij} - C_{-i} - C_{-j} + C_{i} + C_{j}+ S_{tot}$', cbar=True, notext=True)

hp.gnomview(sb_full_open + sb_both_close - sb1_close - sb2_close, sub=(2, 2, 3), rot=(0, 90), reso=5, xsize=350,
            ysize=350,
            title='$S_{-ij} - C_{-i} - C_{-j} + S_{tot}$', cbar=True, notext=True)
