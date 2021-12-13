'''
This script consist in function  to translate angles to coordinate system or angles and reversly.

Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file is part of unsupervised_gaze_calibration.

unsupervised_gaze_calibration is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

unsupervised_gaze_calibration is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with unsupervised_gaze_calibration. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np


def getEulerAngles(R):
    """
    Obtain the Euler angles corresponding to the rotation matrix R
    IMPORTANT: This function assumes all the angles are within the range -90 to 90 degrees!!
    """
    rx = -np.arcsin(R[1, 2])
    ry = np.arctan2(R[0, 2], R[2, 2])
    rz = np.arctan2(R[1, 0], R[1, 1])

    return rz, rx, ry


def getRotationMatrix(rz, rx, ry):
    """
    Returns a rotation matrix from the three euler angles defined for the world coordinate system

    It can be interpreted as a rotation in the direction of z, then in the same coordinate
    system (world) a rotation in the direction of x, and finally a rotation in the direction of y
          rz->roll : rotation in z
          rx->pitch: rotation in x, equivalent to tilt
          ry->yaw  : rotation in y, equivalent to pan
    """
    R= [[np.cos(rz)*np.cos(ry)+np.sin(ry)*np.sin(rx)*np.sin(rz) ,-np.cos(ry)*np.sin(rz)+np.sin(ry)*np.sin(rx)*np.cos(rz), np.sin(ry)*np.cos(rx)],\
        [       np.cos(rx)*np.sin(rz)                           ,            np.cos(rx)*np.cos(rz)                      ,        -np.sin(rx)   ],\
        [-np.sin(ry)*np.cos(rz)+np.cos(ry)*np.sin(rx)*np.sin(rz), np.sin(ry)*np.sin(rz)+np.cos(ry)*np.sin(rx)*np.cos(rz), np.cos(ry)*np.cos(rx)]]
    return np.array(R,np.float32)


def vectorToYawElevation(vector, unit='deg', display=False):
    """ Convert a vector to two angle representation by building the rotation matrix where the input vector is z axis.
        The input vector become the z axis, x axis is kept in the XZ plan of the coordinate system, on the left of the
        z axis. The y axis is deduced from the two firsts.
        The output angles are:
            - the first angle is the euler's yaw, called "yaw" or "phi".
            - the second is the inverse euler's pitch, called "elevation" or "theta") """
    vector = np.resize(vector, (1, 3))[0]

    R = [0.0, 0.0, 0.0]
    R[2] = vector / np.linalg.norm(vector)
    R[0] = np.cross([0.0, 1.0, 0.0], R[2])
    R[1] = np.cross(R[2], R[0])

    roll, pitch, yaw = getEulerAngles(np.array(R).T)
    if unit == 'deg':
        yaw = yaw / np.pi*180
        elevation = -pitch / np.pi*180
    elif unit == 'rad':
        elevation = -pitch
    else:
        raise Exception('Unknow angle unit ' + unit)

    return [yaw, elevation]


def yawElevationToVector(yawElevation, unit='deg', display=False):
    """ Transforms an angular representation of gaze (phi, theta) into a vectorial one (x,y,z).
        WARNING: angles outer than [-90, 90] will give wrong vectors """
    yawElevation = list(map(float, yawElevation))
    if unit == 'deg':
        yaw, elevation = yawElevation[0]/180*np.pi, yawElevation[1]/180*np.pi
        if np.abs(yaw) >= 90 or np.abs(elevation) >= 90:
            raise Exception('Can not handle angles outer [-90, 90]')
    elif unit == 'rad':
        yaw, elevation = yawElevation
        if np.abs(yaw) >= np.pi/2 or np.abs(elevation) >= np.pi/2:
            raise Exception('Can not handle angles outer [-90, 90]')
    else:
        raise Exception('Unknow angle unit ' + unit)

    y = np.sin(elevation)

    x = np.sqrt((np.cos(elevation)**2 * np.tan(yaw)**2) / (1 + np.tan(yaw)**2))
    if yaw < 0:  # The signed of x is taken from the sign of the phi
        x = -x

    if x**2 + y**2 >= 1.0:
        z = 0.0
    else:
        z = np.sqrt(1 - x**2 - y**2)
        
    if display and np.isnan(z):
        print('[yawElevationToVector] input: {} output: {}'.format(np.array(yawElevation).flatten()*180/np.pi, [x, y, z]))

    return np.resize([x, y, z], (3, 1))


def angleBetweenVectors_deg(x, y):
    """ Angle in degree between x and y vectors """
    x = np.resize(x.copy(), (3, 1))
    y = np.resize(y.copy(), (3, 1))
    arg = np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    arg = min(1., arg)  # avoid numerical errors
    angle = np.arccos(arg) / np.pi * 180
    return np.linalg.norm(angle)
