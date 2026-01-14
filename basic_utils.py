"""
基础计算工具函数模块 (Numba JIT 加速)

包含光线追踪所需的核心物理计算函数：
- 体素相位获取
- 随机单位向量生成
- 表面法线计算
- 菲涅尔反射率
- 折射向量计算
- 冰吸收系数计算

所有函数均为模块级函数，兼容 Numba @njit 加速。
"""

import numpy as np
from numba import njit


# ==============================================================================
#  物理常量
# ==============================================================================
N_AIR: float = 1.0      # 空气折射率
N_ICE: float = 1.31     # 冰折射率


# ==============================================================================
#  Numba 加速的核心函数
# ==============================================================================

@njit(fastmath=True, cache=True)
def get_voxel_phase(x: float, y: float, z: float, medium: np.ndarray, nx: int) -> int:
    """
    获取指定坐标处的介质相。
    
    :param x: 指定坐标位置 x (整数体素索引)
    :param y: 指定坐标位置 y (整数体素索引)
    :param z: 指定坐标位置 z (整数体素索引)
    :param medium: 二值化的三维介质
    :param nx: 介质正方体边长

    :return: 1(Ice), 0(Air), -1(超出介质范围)
    """
    ix = int(x)
    iy = int(y)
    iz = int(z)

    if ix < 0 or ix >= nx or iy < 0 or iy >= nx or iz < 0 or iz >= nx:
        return -1  # 超出介质范围

    return 1 if medium[ix, iy, iz] else 0


@njit(fastmath=True, cache=True)
def random_unit_vector_no_trig() -> tuple[float, float, float]:
    """
    生成均匀分布的随机单位向量 (拒绝采样法，极快，无三角函数)
    
    :return: (dx, dy, dz) 单位向量分量
    """
    while True:
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        z = np.random.uniform(-1.0, 1.0)
        r2 = x * x + y * y + z * z
        if 0.0001 < r2 <= 1.0:
            inv_r = 1.0 / np.sqrt(r2)
            return x * inv_r, y * inv_r, z * inv_r


@njit(fastmath=True, cache=True)
def normalize_vector(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    """
    归一化向量，消除浮点累积误差。
    
    :param dx, dy, dz: 向量分量
    :return: 归一化后的 (dx, dy, dz)
    """
    length = np.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-10:
        return 0.0, 0.0, 1.0  # 默认返回 z 轴正方向
    inv_len = 1.0 / length
    return dx * inv_len, dy * inv_len, dz * inv_len


@njit(fastmath=True, cache=True)
def get_surface_normal(x: float, y: float, z: float, medium: np.ndarray) -> tuple[float, float, float]:
    """
    计算体素边界的法线 (指向空气侧)。
    使用中心差分梯度法。
    
    :param x, y, z: 碰撞点坐标 (体素单位)
    :param medium: 二值化的三维介质
    
    :return: (nx, ny, nz) 法线向量，若无法确定返回 (0, 0, 0)
    """
    nx_dim, ny_dim, nz_dim = medium.shape
    ix, iy, iz = int(x), int(y), int(z)
    
    # 限制索引范围，防止越界
    x_L = max(0, ix - 1)
    x_R = min(nx_dim - 1, ix + 1)
    y_L = max(0, iy - 1)
    y_R = min(ny_dim - 1, iy + 1)
    z_L = max(0, iz - 1)
    z_R = min(nz_dim - 1, iz + 1)
    
    # medium: 0(Air), 1(Ice). 梯度方向: Air->Ice
    # 表面法线定义为指向外部(Air)，故取负梯度
    gx = float(get_voxel_phase(x_R, iy, iz, medium, nx_dim)) - float(get_voxel_phase(x_L, iy, iz, medium, nx_dim))
    gy = float(get_voxel_phase(ix, y_R, iz, medium, ny_dim)) - float(get_voxel_phase(ix, y_L, iz, medium, ny_dim))
    gz = float(get_voxel_phase(ix, iy, z_R, medium, nz_dim)) - float(get_voxel_phase(ix, iy, z_L, medium, nz_dim))
    
    norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    if norm < 1e-10:
        return 0.0, 0.0, 0.0  # 无法确定法线，通常发生在平坦区域内部
    
    return -gx / norm, -gy / norm, -gz / norm


@njit(fastmath=True, cache=True)
def fresnel_reflectivity(n1: float, n2: float, cos_i: float) -> float:
    """
    计算非偏振光的菲涅尔反射率 R。
    
    :param n1: 入射介质折射率
    :param n2: 透射介质折射率
    :param cos_i: 入射角余弦 (必须为正值)
    
    :return: 反射率 R (0 ~ 1)
    """
    sin_i = np.sqrt(max(0.0, 1.0 - cos_i ** 2))
    sin_t = (n1 / n2) * sin_i
    
    # 全反射 (TIR)
    if sin_t >= 1.0:
        return 1.0
    
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t ** 2))
    
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    
    return 0.5 * (rs ** 2 + rp ** 2)


@njit(fastmath=True, cache=True)
def refract_vector(dx: float, dy: float, dz: float, 
                   nx: float, ny: float, nz: float, 
                   n1: float, n2: float) -> tuple[float, float, float]:
    """
    计算折射后的向量方向 (Snell's Law 矢量形式)。
    
    :param dx, dy, dz: 入射方向向量 (应为单位向量)
    :param nx, ny, nz: 表面法线向量 (应为单位向量，指向入射侧)
    :param n1: 入射介质折射率
    :param n2: 透射介质折射率
    
    :return: (tx, ty, tz) 折射方向向量，若全反射返回 (0, 0, 0)
    """
    eta = n1 / n2
    # cos_i = - dot(I, N)
    dot_val = dx * nx + dy * ny + dz * nz
    cos_i = -dot_val
    
    sin_t2 = eta ** 2 * (1.0 - cos_i ** 2)
    if sin_t2 > 1.0:
        return 0.0, 0.0, 0.0  # 全反射标识
        
    cos_t = np.sqrt(1.0 - sin_t2)
    # T = eta * I + (eta * cos_i - cos_t) * N
    tx = eta * dx + (eta * cos_i - cos_t) * nx
    ty = eta * dy + (eta * cos_i - cos_t) * ny
    tz = eta * dz + (eta * cos_i - cos_t) * nz
    return tx, ty, tz


# ==============================================================================
#  非 Numba 函数 (用于类调用)
# ==============================================================================

def compute_ice_absorption(wavelength_nm: float, n_imag: float = 1.3e-5) -> float:
    """
    根据波长计算冰的吸收系数。
    
    :param wavelength_nm: 波长 (nm)
    :param n_imag: 冰的折射率虚部
    
    :return: 冰的吸收系数 (m^-1)
    """
    wavelength_m = wavelength_nm * 1e-9  # 转换为米
    ice_absorption = 4.0 * np.pi * n_imag / wavelength_m  # m^-1
    return ice_absorption
