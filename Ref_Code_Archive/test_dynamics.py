"""
test_dynamics.py
test DynamicsTools

1st written by: Wonhee Lee
1st written on: 2023 APR 01
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import DynamicsTools as dT
import numpy as np


# KINEMATICS //////////////////////////////////////////////////////////////////
v = np.array([0, 0, 1])
A = np.eye(3)
x_hom = dT.euclidean_to_homogeneous(A)
print(x_hom)

x = dT.homogenous_to_euclidean(x_hom)
print(x)

R = np.array([[1, 2, 3],[2, 3, 4], [3, 4, 5]])
t = v
print(dT.E_hom(R, t))


# print(B @ v)
# print(B @ v.reshape(-1, 1))
# print(B.shape)
# print(np.zeros((B.shape[1],)))
