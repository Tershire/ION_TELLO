"""
DynamicsTools.py
functions and classes for kinematics and kinetics

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import numpy as np


# KINEMATICS //////////////////////////////////////////////////////////////////
# -----------------------------------------------------------------------------
def hom(x):
    """
    euclidean_to_homogeneous
    """
    if x.ndim == 1:  # if x is vector
        x_hom = np.append(x, 1)

    elif x.ndim == 2:  # if x is matrix
        row = np.hstack((np.zeros((x.shape[1] - 1,)), 1))
        x_hom = np.vstack((x, row))

    return x_hom


def euc(x_hom):
    """
    homogenous_to_euclidean
    """
    if x_hom.ndim == 1:  # if x is vector
        x = np.delete(x_hom, -1)

    elif x_hom.ndim == 2:  # if x is matrix
        x = np.delete(x_hom, (-1), axis=0)

    return x


# -----------------------------------------------------------------------------
def E_hom(R, t):
    """
    homogeneous Euclidean transform matrix
    R: (3, 3) rotation matrix
    t: (3,)   translation vector
    """
    return np.vstack((np.hstack((R, t.reshape(-1, 1))),
                      np.array([0, 0, 0, 1])))


# -----------------------------------------------------------------------------
# passive {x, y, z} rotations
def R_x(ang, unit):
    if unit == 'deg':
        ang = np.deg2rad(ang)
    elif unit == 'rad':
        pass
    else:
        raise Exception("wrong unit: must be 'rad' or 'deg'")
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), np.sin(ang)],
                     [0, -np.sin(ang), np.cos(ang)]])

def R_y(ang, unit):
    if unit == 'deg':
        ang = np.deg2rad(ang)
    elif unit == 'rad':
        pass
    else:
        raise Exception("wrong unit: must be 'rad' or 'deg'")
    return np.array([[np.cos(ang), 0, -np.sin(ang)],
                     [0, 1, 0],
                     [np.sin(ang), 0, np.cos(ang)]])

def R_z(ang, unit):
    if unit == 'deg':
        ang = np.deg2rad(ang)
    elif unit == 'rad':
        pass
    else:
        raise Exception("wrong unit: must be 'rad' or 'deg'")
    return np.array([[np.cos(ang), np.sin(ang), 0],
                     [-np.sin(ang), np.cos(ang), 0],
                     [0, 0, 1]])
