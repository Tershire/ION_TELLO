"""
test_delta_yaw.py
test
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import numpy as np

# SET /////////////////////////////////////////////////////////////////////////
i_b_b = np.array([1, 0, 0])
o_t_b = np.array([0, 0, 0])
k_t_b = np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])

# DEF /////////////////////////////////////////////////////////////////////////
def get_delta_yaw():
    """
    calculate delta yaw to align drone x axis and
    target z axis projected onto drone xy plane
    """
    global o_t_b, k_t_b

    if o_t_b is not None and \
       k_t_b is not None:
        k_t_b_proj = k_t_b[0:2]  # normal of target projected on {x_b, y_b} plane
        print(-k_t_b_proj)
        print(i_b_b[0:2])
        print(np.dot(-k_t_b_proj, i_b_b[0:2]))
        if k_t_b_proj[0] < 0:  # if drone is confronting marker
            psi = np.arccos(np.dot(-k_t_b_proj, i_b_b[0:2]))

            # set yaw direction because psi is [0, pi]
            if k_t_b_proj[1] > 0:
                psi = -psi
            elif k_t_b_proj[1] < 0:
                pass
        else:
            psi = 0

        return psi

# RUN /////////////////////////////////////////////////////////////////////////
psi = get_delta_yaw()
print(psi)
