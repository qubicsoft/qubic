import numpy as np
from matplotlib.pyplot import *
import csv
from scipy import stats

dir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/Calib_mount_az/'


def get_column(directory, name_file, num):
    csv_file = csv.reader(open(directory + name_file))
    column = []
    line = 0
    for row in csv_file:
        print(type(row[num]))
        if line != 0:
            value = float(row[num])
            print(value)
            column.append(value)
        line += 1
        print(line)
    return column


def theta_theo_vs_meas(directory, name_file, stop, ve_speed, degree_step):
    theta_theo = get_column(directory, name_file, 5)
    theta_meas = get_column(directory, name_file, 4)

    # Linear fit
    x = np.arange(-35, 35, 1)
    slope_pos, intercept_pos, _, _, std_err_pos = stats.linregress(theta_theo[:stop], theta_meas[:stop])
    slope_neg, intercept_neg, _, _, std_err_neg = stats.linregress(theta_theo[stop:], theta_meas[stop:])

    figure('fit_VE{0}_{1}deg'.format(ve_speed, degree_step))
    subplot(211)
    plot(theta_theo[:stop], theta_meas[:stop], 'ro', label='measurements')
    plot(x, slope_pos * x + intercept_pos, 'b',
         label='linear fit \n a={0} \n b={1} \n std_err={2}'.format(slope_pos, intercept_pos, std_err_pos))
    xlabel(r'$\theta$ theoretical (degree)')
    ylabel(r'$\theta$ measured (degree)')
    legend(numpoints=1, loc='best')
    title('VE{0}_{1}deg positiv rotation'.format(ve_speed, degree_step))

    subplot(212)
    plot(theta_theo[stop:], theta_meas[stop:], 'ro', label='measurements')
    plot(x, slope_neg * x + intercept_neg, 'b',
         label='linear fit \n a={0} \n b={1} \n std_err={2}'.format(slope_neg, intercept_neg, std_err_neg))
    xlabel(r'$\theta$ theoretical (degree)')
    ylabel(r'$\theta$ measured (degree)')
    legend(numpoints=1, loc='best')
    title('VE{0}_{1}deg negativ rotation'.format(ve_speed, degree_step))

    return


def diff_theta(directory, name_file, stop, ve_speed, degree_step):
    theta_theo = np.array(get_column(directory, name_file, 5))
    theta_meas = np.array(get_column(directory, name_file, 4))

    diff_pos = (theta_meas[:stop] - theta_theo[:stop]) # / theta_theo[:stop]
    diff_neg = (theta_meas[stop:] - theta_theo[stop:]) # / theta_theo[stop:]

    # Linear fit
    x = np.arange(-35, 35, 1)
    slope_pos, intercept_pos, _, _, std_err_pos = stats.linregress(theta_theo[:stop], diff_pos)
    slope_neg, intercept_neg, _, _, std_err_neg = stats.linregress(theta_theo[stop:], diff_neg)

    figure('fit_diff_VE{0}_{1}deg'.format(ve_speed, degree_step))
    subplot(211)
    plot(theta_theo[:stop], diff_pos, 'ro', label='measurements')
    plot(x, slope_pos * x + intercept_pos, 'b',
         label='linear fit \n a={0} \n b={1} \n std_err={2}'.format(slope_pos, intercept_pos, std_err_pos))
    xlabel(r'$\theta_{theo}$ (degree)')
    ylabel(r'$\theta_{meas} - \theta_{theo}$ (degree)')
    legend(numpoints=1, loc='best')
    title('VE{0}_{1}deg positiv rotation'.format(ve_speed, degree_step))

    subplot(212)
    plot(theta_theo[stop:], diff_neg, 'ro', label='measurements')
    plot(x, slope_neg * x + intercept_neg, 'b',
         label='linear fit \n a={0} \n b={1} \n std_err={2}'.format(slope_neg, intercept_neg, std_err_neg))
    xlabel(r'$\theta_{theo}$ (degree)')
    ylabel(r'$\theta_{meas} - \theta_{theo}$ (degree)')
    legend(numpoints=1, loc='best')
    title('VE{0}_{1}deg negativ rotation'.format(ve_speed, degree_step))

    return np.mean(diff_pos)/degree_step, np.mean(diff_neg)/degree_step


### Histogram
delta_theta_meas_VE1_2deg = get_column(dir, 'calib_mount_az_VE1_2deg.csv', 7)

figure('hist_VE1_2deg')
hist(delta_theta_meas_VE1_2deg[:31], bins=15)
xlabel(r'$\Delta\theta$ obtained')
title('Histogram for VE1, 2$\degree$ asked')

figure('hist_VE1_-2deg')
hist(delta_theta_meas_VE1_2deg[31:], bins=15)
xlabel(r'$\Delta\theta$ obtained')
title('Histogram for VE1, -2$\degree$ asked')

delta_theta_meas_VE1_5deg = get_column(dir, 'calib_mount_az_VE1_5deg.csv', 7)

figure('hist_VE1_5deg')
hist(delta_theta_meas_VE1_5deg[:12], bins=30)
xlabel(r'$\Delta\theta$ obtained')
title('Histogram for VE1, 5$\degree$ asked')

figure('hist_VE1_-5deg')
hist(delta_theta_meas_VE1_5deg[12:], bins=30)
xlabel(r'$\Delta\theta$ obtained')
title('Histogram for VE1, -5$\degree$ asked')

### Theta theo and measured for one file
theta_theo_VE1_2deg = get_column(dir, 'calib_mount_az_VE1_2deg.csv', 5)
theta_meas_VE1_2deg = get_column(dir, 'calib_mount_az_VE1_2deg.csv', 4)

figure('theta_VE1_2deg')
plot(theta_meas_VE1_2deg, 'r+', label=r'$\theta$ meas')
plot(theta_theo_VE1_2deg, 'b+', label=r'$\theta$ theo')
legend(numpoints=1)
xlabel('measurement index')
ylabel(r'$\theta$ (degrees)')

### Plot theta theo vs theta meas and fit it
theta_theo_vs_meas(dir, 'calib_mount_az_VE1_2deg.csv', 31, 1, 2)
theta_theo_vs_meas(dir, 'calib_mount_az_VE1_5deg.csv', 12, 1, 5)
theta_theo_vs_meas(dir, 'calib_mount_az_VE1_10deg.csv', 6, 1, 10)
theta_theo_vs_meas(dir, 'calib_mount_az_VE6_10deg.csv', 6, 6, 10)

### Plot the diff
diff_pos, diff_neg = diff_theta(dir, 'calib_mount_az_VE1_2deg.csv', 31, 1, 2)
diff_pos, diff_neg = diff_theta(dir, 'calib_mount_az_VE1_5deg.csv', 12, 1, 5)
diff_pos, diff_neg = diff_theta(dir, 'calib_mount_az_VE1_10deg.csv', 6, 1, 10)
diff_pos, diff_neg = diff_theta(dir, 'calib_mount_az_VE6_10deg.csv', 6, 6, 10)
