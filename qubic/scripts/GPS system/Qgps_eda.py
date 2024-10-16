#all the codes

# General packages
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import datetime as dt
from datetime import timedelta
import matplotlib as mpl

mpl.rcParams['lines.markersize'] = 0

def read_gps_bindat(filename):
    '''
    read the binary data acquired from by RTK simple broadcast
    '''
    if not os.path.isfile(filename):
        print('ERROR!  File not found: %s' % filename)
        return

    # read the data
    h = open(filename,'rb')
    bindat = h.read()
    h.close()

    # interpret the binary data
    fmt = '<Bdiiiiiiifi'
    nbytes = 45
    names = "STX,timestamp,rpN,rpE,rpD,roll,yaw,pitchIMU,rollIMU,temperature,checksum".split(',')
    data = {}
    for name in names:
        data[name] = []    

    idx = 0
    while idx+nbytes<len(bindat):
        packet = bindat[idx:idx+nbytes]
        dat_list = struct.unpack(fmt,packet)

        if len(dat_list)!=len(names):
            print('ERROR:  Incompatible data.')
            return data

        for datidx,name in enumerate(names):
            data[name].append(dat_list[datidx])

        idx += nbytes

    return data

#dat = read_gps_bindat('calsource_orientation.dat')

def create_date_axis(data_file):
    dat = read_gps_bindat(data_file)

    date_axis = []
    for tstamp in dat['timestamp']:
        date=dt.datetime.utcfromtimestamp(tstamp)
        date+= dt.timedelta(hours=2) #make date according to Paris local hour UTC+2
        date_axis.append(date)
    return date_axis

def conversion(dat):
    dat['rpN'] = [value / 10000 for value in dat['rpN']]
    dat['rpE'] = [value / 10000 for value in dat['rpE']]
    dat['rpD'] = [value / 10000 for value in dat['rpD']]
    dat['roll'] = [value / 1000 for value in dat['roll']]
    dat['yaw'] = [value / 1000 for value in dat['yaw']]
    return dat

def date_to_indice(date,data_file):

    # date : string ('year-month-dayThour:minute:second') or isoformat
    # data : list
    date_axis=create_date_axis(data_file)
    if type(date) is str:
        date = dt.datetime.fromisoformat(date)
    for i in range(0,len(date_axis)):
        if date_axis[i]== date:
           return(i)
    return('You have not taken data at this date')


#components_fig : creates a figure with the north, east and down components and the roll and yaw angles as a function of time
def components_fig(start_date, end_date,data_file):
        # start_date : string ('year-month-dayThour:minute:second')
        #end_date : string ('year-month-dayThour:minute:second')
        # dat: DATA File
        
        #put the table values ​​in m and degrees
        dat = read_gps_bindat(data_file)
        date_axis=create_date_axis(data_file)
        dat=conversion(dat)

        
        #index associated with these dates
        start_index=date_to_indice(start_date,data_file)
        end_index=date_to_indice(end_date,data_file)
        
        #in case of date error
        if type(start_index) is str or type(end_index) is str:
            print('You have not taken data at this date verify your start date and your end date')
            return False
        
        #colors
        color_r = 'tab:red'
        color_b = 'tab:blue'
        color_d = 'tab:green'
        color_p = 'tab:pink'
        color_v = 'tab:purple'
          
        #create plot
        fig, ax1 = plt.subplots(figsize = (15,5))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('North East & Down Component (m)', color = color_r)
        ax1.plot(date_axis[start_index:end_index], dat['rpN'][start_index:end_index],'.',markersize=1, color = color_r, label = 'North component')
        ax1.plot(date_axis[start_index:end_index], dat['rpE'][start_index:end_index],'.',markersize=1, color = color_b, label = 'East component')
        ax1.plot(date_axis[start_index:end_index], dat['rpD'][start_index:end_index],'.',markersize=1, color = color_d, label = 'Down component')
        #ax1.set_ylim(lower_limite_,upper_limit)
        ax2 = ax1.twinx()


        ax2.set_xlabel('Date')
        ax2.set_ylabel('roll & yaw angle (degree)', color = color_b)
        ax2.plot(date_axis[start_index:end_index], dat['roll'][start_index:end_index],'.',markersize=1, color = color_v,  label = 'roll')
        ax2.plot(date_axis[start_index:end_index], dat['yaw'][start_index:end_index],'.',markersize=1, color = color_p, label = 'yaw')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.set_title(date_axis[start_index])
        #ax2.set_ylim(lower_limite_,upper_limit)
        fig.legend()
        plt.show()




#components_values : print the north, east and down components and the roll and yaw angles and return an array of the components values at this time
def componnent_values(date,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    index=date_to_indice(date,data_file)  
    if type(index) is str:
        print('You have not taken data at this date')
        return False
    #print(f"rpN={dat['rpN'][index]}, rpE={dat['rpE'][index]}, rpD={dat['rpD'][index]}, roll={dat['roll'][index]},yaw={dat['yaw'][index]}")
    return np.array([dat['rpN'][index],dat['rpE'][index],dat['rpD'][index],dat['roll'][index],dat['yaw'][index]])


#calculates the average of the components between 2 instants
def components_average(start_date, end_date,data_file):
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    start_index=date_to_indice(start_date,data_file)
    end_index=date_to_indice(end_date,data_file)
    east_average=np.mean(dat['rpE'][start_index:end_index])
    north_average=np.mean(dat['rpN'][start_index:end_index])
    down_average=np.mean(dat['rpD'][start_index:end_index])
    roll_average=np.mean(dat['roll'][start_index:end_index])
    yaw_average=np.mean(dat['yaw'][start_index:end_index])
    return np.array([north_average,east_average,down_average,roll_average,yaw_average])


#noise calculate the standard deviation of rpN,rpE and rpD data set; print the values and return an array of values
def noise(start_date, end_date,data_file):
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    start_index=date_to_indice(start_date,data_file)
    end_index=date_to_indice(end_date,data_file)
    east_noise=np.std(dat['rpE'][start_index:end_index])
    north_noise=np.std(dat['rpN'][start_index:end_index])
    down_noise=np.std(dat['rpD'][start_index:end_index])
    #print(f"the standard deviation of : north={north_noise}, east={east_noise} and down={down_noise}")
    return np.array([north_noise,east_noise,down_noise])

#noise calculate the standard deviation of roll an yaw data set; print the values and return an array of values
def noise_angle(start_date, end_date,data_file):
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    start_index=date_to_indice(start_date,data_file)
    end_index=date_to_indice(end_date,data_file)
    roll_noise=np.std(dat['roll'][start_index:end_index])
    yaw_noise=np.std(dat['yaw'][start_index:end_index])
    #print(f"the standard deviation of : roll={roll_noise} and yaw={yaw_noise}")
    return np.array([roll_noise,yaw_noise])

# I am not sure about this fonction, I use course formula but you have to check that I have applied it correctly
def dispersion(start_date, end_date, distance_antenna, data_file):
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    noise_components=noise(start_date, end_date,data_file)
    noise_roll_yaw=noise_angle(start_date, end_date,data_file)
    #mean_component=np.array([rpN_average,rpE_average,rpD_average,roll_average,yaw_average])
    average=components_average(start_date, end_date,data_file)
    
    dispersion_atenna_1=noise_components/2
    dispersion_antenna_2_north=np.sqrt((distance_antenna*np.cos(np.radians(average[3]))*np.cos(np.radians(average[4]+(np.pi/2)))*noise_roll_yaw[1])**2+(distance_antenna*np.sin(np.radians(average[3]))*np.sin(np.radians(average[4]+(np.pi/2)))*noise_roll_yaw[0])**2+(noise_components[0])**2)
    dispersion_antenna_2_east=np.sqrt((distance_antenna*np.sin(np.radians(average[3]))*np.cos(np.radians(average[4]+(np.pi/2)))*noise_roll_yaw[1])**2+(distance_antenna*np.sin(np.radians(average[3]))*np.cos(np.radians(average[4]+(np.pi/2)))*noise_roll_yaw[0])**2+(noise_components[1])**2)
    dispersion_antenna_2_down=np.sqrt((distance_antenna*np.sin(np.radians(average[4]+(np.pi/2)))*noise_roll_yaw[1])**2+(noise_components[2])**2)
    dispersion_atenna_2=np.array([dispersion_antenna_2_north,dispersion_antenna_2_east,dispersion_antenna_2_down])
    return dispersion_atenna_1,dispersion_atenna_2



#if we do a study of noise as a function of distance,this function allows you to create a data set for the standard deviation depending on the distance
def noise_data(start_date_list, end_date_list,distance_list,data_file):
    # start_date_list :list of string ('year-month-dayThour:minute:second')
    #end_date_list :list of string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_list : distances between the antenna (m)
    #!!! len of start_date_list, end_date_list and distance_list must be the same 
    
    if len(start_date_list)==len(end_date_list) and len(start_date_list)==len(distance_list):
        print(' len of start_date_list, end_date_list and distance_list must be the same ')
        return False
    noise_data=[]
    for i in range (0,len(start_date_list)):
        noise_result = noise(start_date_list[i], end_date_list[i],data_file)
        noise_data.append([noise_result[0], noise_result[1], noise_result[2], distance_list[i]])
    return noise_data
    
def noise_data_angle(start_date_list, end_date_list,distance_list,data_file):
    # start_date_list :list of string ('year-month-dayThour:minute:second')
    #end_date_list :list of string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_list : distances between the antenna (m)
    #!!! len of start_date_list, end_date_list and distance_list must be the same 
    
    if len(start_date_list)==len(end_date_list) and len(start_date_list)==len(distance_list):
        print(' len of start_date_list, end_date_list and distance_list must be the same ')
        return False
    noise_data_angle=[]
    
    for i in range (0,len(start_date_list)):
        noise_result=noise_angle(start_date_list[i], end_date_list[i],data_file)
        noise_data_angle.append([noise_result[0], noise_result[1], distance_list[i]])
    return noise_data_angle


# create a plot of the noise in function of the distances
def noise_curve( noise_data, noise_data_angle):

  plt.figure()
  plt.plot(noise_data[:,3], noise_data[:,0], label='east noise')
  plt.plot(noise_data[:,3], noise_data[:,1], label='north noise')
  plt.plot(noise_data[:,3], noise_data[:,2], label='down noise')
  plt.plot(noise_data_angle[:,2], noise_data_angle[:,0], label='roll noise')
  plt.plot(noise_data_angle[:,2], noise_data_angle[:,1], label='yaw noise')
  plt.xlabel("Antenna Distance")
  plt.ylabel("Standard Deviation of componants")
  plt.title(" Noise Curve")
  plt.legend()
  plt.show()

#vector1_2 : create the vector between the 2 antennas
def vector1_2(date,distance_antenna,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_antenna : distance between the antennas (m)
    
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    date_index=date_to_indice(date,data_file)
    roll=np.radians(dat['roll'][date_index])
    yaw=np.radians(dat['yaw'][date_index])
    d_rpN=distance_antenna*np.cos(roll)*np.sin(yaw+(np.pi/2))
    d_rpE=distance_antenna*np.sin(roll)*np.sin(yaw+(np.pi/2))
    d_rpD=-distance_antenna*np.cos(yaw+(np.pi/2))
    return np.array([d_rpN,d_rpE,d_rpD])

def position_antenna2(date,distance_antenna,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_antenna : distance between the antennas (m)
    dat = read_gps_bindat(data_file)
    date_index=date_to_indice(date,data_file)
    vector=vector1_2(date,distance_antenna,data_file)
    #put the table values ​​in m and degrees
    dat=conversion(dat)
    rpN_2=vector[0]+dat['rpN'][date_index]
    rpE_2=vector[1]+dat['rpE'][date_index]
    rpD_2=vector[2]-dat['rpD'][date_index]
    return np.array([rpN_2,rpE_2,rpD_2])

def position_source(date,ds1_north,ds1_east,ds1_down,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #ds1_north : distance between source and antenna 1 project to north axis (m)
    #ds1_east : distance between source and antenna 1 project to east axis (m)
    #ds1_down : distance between source and antenna 1 project to down axis (m)
    
    #put the table values ​​in m and degrees
    dat = read_gps_bindat(data_file)
    dat=conversion(dat)
    date_index=date_to_indice(date,data_file)
    rpN_S=ds1_north+dat['rpN'][date_index]
    rpE_S=ds1_east+dat['rpE'][date_index]
    rpD_S=ds1_down+dat['rpD'][date_index]
    return np.array([rpN_S,rpE_S,rpD_S])


def orientation(date,distance_antenna,angle_theta,angle_phi,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_antenna : distance between the antennas (m)
    #angle_theta : angle vertical between the vector between the 2 antennas and source degree
    #angle_phi : angle horizontal between the vector between the 2 antennas and source  degree
    
    vector=vector1_2(date,distance_antenna,data_file)
    rpN1_2=vector[0]
    rpE1_2=vector[1]
    rpD1_2=vector[2]
    #-> spherical coordinate
    #ro=np.sqrt(rpN1_2**2+rpE1_2**2+rpD1_2**2)
    theta=np.arctan(rpE1_2/rpN1_2)
    phi=np.arccos(rpD1_2/np.sqrt(rpN1_2**2+rpE1_2**2+rpD1_2**2))
    theta+=np.radians(angle_theta)
    phi+=np.radians(angle_phi)
    #-> Cartesinian coordinate normalised vector 
    rpNp=np.sin(phi)*np.cos(theta)
    rpEp=np.sin(phi)*np.sin(theta)
    rpDp=np.cos(phi)
    return np.array([rpNp,rpEp,rpDp])

def plot_position(date,distance_antenna,data_file):
    #date : string ('year-month-dayThour:minute:second')
    # dat: DATA File
    #distance_antenna : distance between the antennas (m)
    dat = read_gps_bindat(data_file)
    date_index=date_to_indice(date,data_file)
    coordinates_antenna2=position_antenna2(date,distance_antenna,data_file)
    rpN_2,rpE_2,rpD_2=coordinates_antenna2[0],coordinates_antenna2[1],coordinates_antenna2[2]
    # Create the plot
    fig = plt.figure(figsize=(15, 5))
    #fig, ax = plt.subplots(, subplot_kw=dict(projection='3d'))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with size and color customization (optional)
    ax.plot(rpN_2,rpE_2,rpD_2, '.',markersize=20,label='antenne2') 
    ax.plot(dat['rpN'][date_index],dat['rpE'][date_index] , dat['rpD'][date_index],'.',markersize=20,label='antenne1') 
    ax.set_title(date)
    ax.set_xlabel('north')
    ax.set_ylabel('east')
    ax.set_zlabel('down')
    #ax.set_xlim(lower_limite_,upper_limit)
    #ax.set_ylim(lower_limite_,upper_limit)
    #ax.set_zlim(lower_limite_,upper_limit)
    ax.legend()
    plt.show()


def mean_position_antenna2(start_date, end_date,distance_antenna,data_file):
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    #distance_antenna : distance between the antennas (m)
    
    #mean_component=np.array([rpN_average,rpE_average,rpD_average,roll_average,yaw_average])
    
    mean_components=components_average(start_date, end_date,data_file)
    roll=np.radians(mean_components[3])
    yaw=np.radians(mean_components[2])
    rpN2=distance_antenna*np.cos(roll)*np.sin(yaw-(np.pi/2))
    rpE2=distance_antenna*np.sin(roll)*np.sin(yaw-(np.pi/2))
    rpD2=-distance_antenna*np.cos(yaw+(np.pi/2))
    rpN2+=mean_components[0]
    rpE2+=mean_components[1]
    rpD2-=mean_components[2]
    return np.array([rpN2,rpE2,rpD2])


def plot_pos_moy(start_date, end_date,distance_antenna,data_file):
    
    # start_date : string ('year-month-dayThour:minute:second')
    #end_date : string ('year-month-dayThour:minute:second')
    # dat: DATA File 
    #distance_antenna : distance between the antennas (m)
    
    #mean_component=np.array([rpN_average,rpE_average,rpD_average,roll_average,yaw_average])
    mean_components=components_average(start_date, end_date,data_file)
    mean_position2=mean_position_antenna2(start_date, end_date,distance_antenna,data_file)
    rpN2,rpE2,rpD2=mean_position2[0],mean_position2[1],mean_position2[2]
    fig = plt.figure(figsize=(15, 5))
    #fig, ax = plt.subplots(, subplot_kw=dict(projection='3d'))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with size and color customization (optional)
    ax.plot(rpN2,rpE2,rpD2, '.',markersize=20,label='antenne2') 
    ax.plot(mean_components[0],mean_components[1],mean_components[2],'.',markersize=20,label='antenne1')
    ax.set_title(f"position moyen de {start_date} à {end_date}")
    ax.set_xlabel('north')
    ax.set_ylabel('east')
    ax.set_zlabel('down')
    #ax.set_xlim(lower_limite_,upper_limit)
    #ax.set_ylim(lower_limite_,upper_limit)
    #ax.set_zlim(lower_limite_,upper_limit)
    ax.legend()
    plt.show()

#value for example
distance_antenna=2
ds1_north,ds1_east,ds1_down=1,1,1
angle_theta,angle_phi=90,0

def calsource_info(data_file):
    # dat: DATA File sting 'name_data'

    global distance_antenna,ds1_north,ds1_east,ds1_down,angle_theta,angle_phi
    dat = read_gps_bindat(data_file)

    list(dat)
    date_axis=create_date_axis(data_file)
    start_date=date_axis[0]
    end_date=date_axis[-1]
    components_fig(start_date, end_date ,data_file)
    
    position_of_antenna1=[]
    position_of_antenna2=[]
    position_of_source=[]
    orientation_of_source=[]
    for i in range(0, len(dat['timestamp'])):
        date = start_date + dt.timedelta(seconds=(0.125 * i))
        index = date_to_indice(date,data_file)
        
        if isinstance(index, int):
            position2 = position_antenna2(date, distance_antenna, data_file)
            orientation_source = orientation(date, distance_antenna, angle_theta, angle_phi, data_file)
            positions = position_source(date, ds1_north, ds1_east, ds1_down,data_file)
            position1 = [dat['rpN'][index], dat['rpE'][index], dat['rpD'][index]]
            
            '''print(f"position of antenna 1 is north component = {dat['rpN'][index]}, east component = {dat['rpE'][index]} and down component = {dat['rpD'][index]} ")
            print(f"position of antenna 2 is north component = {position2[0]}, east component = {position2[1]} and down component = {position2[2]}")
            print(f"position of the source is north component = {positions[0]}, east component = {positions[1]} and down component = {positions[2]}")
            print(f"the orientation of the source is north direction = {orientation_source[0]}, east direction = {orientation_source[1]} and down direction = {orientation_source[2]}")
            print(' ')'''
            position_of_antenna1.append(position1)
            position_of_antenna2.append(position2)
            position_of_source.append(positions)
            orientation_of_source.append(orientation_source)
    
    return position_of_antenna1, position_of_antenna2, position_of_source, orientation_of_source

position_of_antenna1, position_of_antenna2, position_of_source, orientation_of_source = calsource_info('calsource_orientation.dat')
