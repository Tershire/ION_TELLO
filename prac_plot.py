import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# https://stackoverflow.com/questions/27023068/plotting-3d-vectors-using-python-matplotlib

# plot ------------------------------------------------------------------------
# plt.ion()   # interactive mode on

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plot origin: camera (0, 0, 0)
x, y, z = 0, 0, 0
u, v, w = 0, 0, 1

ax.quiver(x, y, z, u, v, w, color='k')

ax.set_xlim([-1, +1])
ax.set_ylim([-0.5, +0.5])
ax.set_zlim([-10, +10])

ax.set_proj_type('persp')
ax.view_init(elev=+90, azim=-90)

# ax.set_aspect('equal')

plt.show()