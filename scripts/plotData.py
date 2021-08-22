#!/usr/bin/python
import os
import struct
import matplotlib.pyplot as plot
import numpy as np
from loadAedatlv import load_aedatlv
from loadNavMat import load_nav_mat
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    filename = 'exp200'
    dirpath = '/home/eneftci/Downloads/2020_05_28/'
    INS_Info, ZUPT_flags, IMU_readouts, velocity, trajectory, heading, timestamp = load_nav_mat(dirpath, filename)

    plot.close()

    # plot accelerometer measurements
    plot.figure(1)
    plot.plot(timestamp, IMU_readouts[0, :])
    plot.grid
    plot.plot(timestamp, IMU_readouts[1, :])
    plot.plot(timestamp, IMU_readouts[2, :])
    ttlMsg = 'Accelerometer readouts'
    plot.title(ttlMsg)
    plot.xlabel('Time, s')
    plot.ylabel('Acceleration, m/s^2')
    plot.legend(['1', '2', '3'])
    # plot.show()

    # plot gyroscope measurements
    plot.figure(2)
    plot.plot(timestamp, IMU_readouts[3, :])
    plot.grid
    plot.plot(timestamp, IMU_readouts[4, :])
    plot.plot(timestamp, IMU_readouts[5, :])
    ttlMsg = 'Gyroscope readouts'
    plot.title(ttlMsg)
    plot.xlabel('Time, s')
    plot.ylabel('Angular rate, deg/s')
    plot.legend(['1', '2', '3'])
    # plot.show()

    # plot velocity estimated by INS
    plot.figure(3)
    yL = ['(1)', '(2)', '(3)']
    plot.subplot(4, 1, 1)
    plot.plot(timestamp, ZUPT_flags, 'r')
    ttlMsg = 'ZUPT state' #Use this one as target
    plot.title(ttlMsg)
    for i in range(3):
        plot.subplot(4, 1, i + 2)
        plot.plot(timestamp, velocity[i, :], 'b')
        plot.grid

        if (i == 0):
            plot.legend(['est left'])
            ttlMsg = 'Velocity NED, m/s.'
            plot.title(ttlMsg)
        plot.ylabel(yL[i])
    plot.xlabel('time, sec.')

    # plot orientation estimated by INS
    yL = ['roll', 'pitch', 'azim.']
    plot.figure(4)
    for i in range(3):
        plot.subplot(3, 1, i + 1)
        plot.plot(timestamp, heading[i, :], 'b')
        plot.grid
        if (i == 1):
            ttlMsg = 'Roll,Pitch,Azim., deg.'
            plot.title(ttlMsg)
        plot.ylabel(yL[i])
    plot.xlabel('time, sec.')

    # plot estimated trajectory
    plot.figure(5)
    plot.plot(trajectory[1, :], trajectory[0, :], 'b')
    plot.plot(trajectory[1, 0], trajectory[0, 0], 's')
    plot.plot(trajectory[1, -1], trajectory[0, -1], '*')
    plot.xlabel('Easting, m')
    plot.ylabel('Northing, m')
    plot.ttlMsg = 'Estimated and True Path, Northing-Easting, m'
    plot.legend(['path', 'begin', 'end'])
    plot.title(ttlMsg)

    #plot.show()

    # plot events
    print('loading events...')
    p, x, y, allTs = load_aedatlv(dirpath, filename)

    x = np.array(x)
    y = np.array(y)
    allTs = np.array(allTs)

    # remove outlier
    x = x[allTs != 0]
    y = y[allTs != 0]
    allTs = allTs[allTs != 0]

    x = x.tolist()
    y = y.tolist()
    allTs = allTs.tolist()

    # plot .aedatlv data
    time_stamps = [(x - allTs[0]) * 10 ** -6 for x in allTs]
    fig = plot.figure(6)
    ax = fig.gca(projection='3d')
    t_of_interest = 30
    k_candidate = np.where(abs(np.array(time_stamps) - t_of_interest) < 0.001)
    k = k_candidate[0][0]
    ax.scatter(time_stamps[k:k + 2000], y[k:k + 2000], x[k:k + 2000], c='b', s=2)
    # plot whole datat set (not recommended since it will take a long time to plot the whole dataset and might run out of memory)
    # ax.scatter(time_stamps[0:-1], x[0:-1], y[0:-1],c='r',s=2)
    ax.set_title('Raw events')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlabel('time, s')
    ax.grid(True)
    plot.show()
