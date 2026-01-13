import numpy as np
from numba import njit, prange

# ==========================================
# 1. Numba 加速内核 (Ray Tracing Core)
# ==========================================

@njit(fastmath=True)
def _get_voxel_phase(x, y, z, medium):
    '''
    获取指定坐标处的介质相。
    
    
    :param x: 指定坐标位置x
    :param y: 指定坐标位置y
    :param z: 指定坐标位置z
    :param medium: 二值化的三维介质

    :return: 1(Ice), 0(Air), -1(超出介质范围)
    '''