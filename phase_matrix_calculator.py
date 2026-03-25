"""
相矩阵计算器 (Phase Matrix Calculator)

基于 Xiong et al., "A New Hybrid Snow Light Scattering Model Based on Geometric
Optics Theory and Vector Radiative Transfer Theory," IEEE TGRS, 2015, Section II-C。

物理原理
--------
在几何光学极限下，雪颗粒尺寸远大于波长，散射仅由冰-空气界面处的菲涅尔反射
和折射决定（忽略衍射和干涉）。对于双连续介质，通过蒙特卡洛方法统计光线与界面
的第一次交互，得到 1-2 参考系下的相矩阵元素 P11(Θ) 和 P22(Θ)。

1-2 参考系定义（论文 Fig. 2 右图）：
  - ê₁ 方向：垂直于散射平面（入射方向 k̂ᵢ 与散射方向 k̂ₛ 所确定的平面）
  - ê₂ 方向：在散射平面内，垂直于 k̂ₛ
  - ê₁ 对应 s 偏振（TE），ê₂ 对应 p 偏振（TM）
  - 因菲涅尔效应不混合偏振：P₁₂ = P₂₁ = 0

散射系数与相矩阵的关系（论文式 (10)/(16)）：
  κ_s = π ∫₀^π [P₁₁(Θ) + P₂₂(Θ)] sin(Θ) dΘ
      = π Σᵢ [P₁₁(Θᵢ) + P₂₂(Θᵢ)] sin(Θᵢ) ΔΘᵢ

归一化方案（论文式 (17)）：
  通过 κ_s 将蒙特卡洛统计得到的相对权重转换为绝对相矩阵值：
  P₁₁(Θᵢ) = acc_P11[i] · κ_s / normalization
  P₂₂(Θᵢ) = acc_P22[i] · κ_s / normalization
  其中 normalization = π Σⱼ (acc_P11[j] + acc_P22[j]) sin(Θⱼ) ΔΘ

算法步骤
--------
1. 随机起点 (x₀, y₀, z₀) 与随机方向 k̂ᵢ
2. DDA 算法追踪光线至第一个冰-空气或空气-冰界面
3. 计算界面法线 n̂（中心差分梯度，指向空气侧）
4. 计算入射角余弦：cos θᵢ = -k̂ᵢ · n̂（法线调整为指向入射侧后取负）
5. 偏振菲涅尔系数：R_s, R_p；透射率：T_s = 1-R_s，T_p = 1-R_p
6. 反射方向：k̂ᵣ = k̂ᵢ - 2(k̂ᵢ·n̂)n̂；散射角：Θᵣ = arccos(k̂ᵢ·k̂ᵣ)
7. 折射方向（Snell 矢量形式）；散射角：Θₜ = arccos(k̂ᵢ·k̂ₜ)
8. 分别将 (R_s, R_p) 和 (T_s, T_p) 累加到对应角度区间
9. 汇总后按式 (17) 归一化为 P₁₁、P₂₂

参考文献
--------
Xiong, C., Shi, J., Ji, D., Wang, T., Xu, Y., & Zhao, T. (2015).
A New Hybrid Snow Light Scattering Model Based on Geometric Optics Theory
and Vector Radiative Transfer Theory.
IEEE Transactions on Geoscience and Remote Sensing, 53(9), 4862-4875.
"""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from basic_utils import (
    N_AIR,
    N_ICE,
    get_voxel_phase,
    get_surface_normal,
    normalize_vector,
    random_unit_vector_no_trig,
    refract_vector,
)


# ==============================================================================
#  命名常量
# ==============================================================================
_DIR_EPSILON: float = 1e-20   # 方向分量零保护（避免 DDA 除以零）
_EPSILON: float = 1e-5         # 浮点偏移量（避免光线卡在界面上）
_MAX_SEGMENT_VOXELS: float = 200.0  # DDA 单段最大步长（体素数），防止无界漫游


# ==============================================================================
#  Numba 加速辅助函数
# ==============================================================================

@njit(fastmath=True, cache=True)
def _fresnel_polarized(
    cos_i: float,
    n1: float,
    n2: float,
) -> Tuple[float, float, float, float, bool]:
    """
    计算 s 和 p 两种偏振分量的菲涅尔反射率与透射率。

    约定与论文 1-2 参考系对齐：
      - s 偏振（ê₁）垂直于散射平面，对应 Fresnel r_s
      - p 偏振（ê₂）在散射平面内，对应 Fresnel r_p

    :param cos_i: 入射角余弦 cos θᵢ（正值，即 θᵢ ∈ [0°, 90°]）
    :param n1: 入射介质折射率
    :param n2: 透射介质折射率

    :return: (R_s, R_p, T_s, T_p, is_tir)
        R_s  : s 偏振强度反射率 = |r_s|²
        R_p  : p 偏振强度反射率 = |r_p|²
        T_s  : s 偏振强度透射率 = 1 - R_s（能量守恒）
        T_p  : p 偏振强度透射率 = 1 - R_p
        is_tir: 是否发生全内反射（True 时 T_s = T_p = 0）
    """
    sin_i_sq = max(0.0, 1.0 - cos_i * cos_i)
    sin_t_sq = (n1 / n2) ** 2 * sin_i_sq

    # 全内反射判断
    if sin_t_sq >= 1.0:
        return 1.0, 1.0, 0.0, 0.0, True

    cos_t = np.sqrt(max(0.0, 1.0 - sin_t_sq))

    # Fresnel 振幅系数（与 basic_utils.fresnel_reflectivity 保持一致的符号约定）
    denom_s = n1 * cos_i + n2 * cos_t
    r_s = (n1 * cos_i - n2 * cos_t) / denom_s if denom_s > 1e-15 else 0.0

    denom_p = n1 * cos_t + n2 * cos_i
    r_p = (n1 * cos_t - n2 * cos_i) / denom_p if denom_p > 1e-15 else 0.0

    R_s = r_s * r_s
    R_p = r_p * r_p
    T_s = 1.0 - R_s
    T_p = 1.0 - R_p

    return R_s, R_p, T_s, T_p, False


@njit(fastmath=True, cache=True)
def _reflect_direction(
    dx: float, dy: float, dz: float,
    nx: float, ny: float, nz: float,
    dot_dn: float,
) -> Tuple[float, float, float]:
    """
    计算反射方向向量。

    反射公式：k̂ᵣ = k̂ᵢ - 2(k̂ᵢ·n̂)n̂

    :param dx, dy, dz: 入射单位方向向量 k̂ᵢ
    :param nx, ny, nz: 表面法线 n̂（已调整为指向入射侧，dot_dn < 0）
    :param dot_dn: 预计算的 k̂ᵢ · n̂（负值）

    :return: (rx, ry, rz) 反射方向（未归一化，但数值上近似为单位向量）
    """
    two_dot = 2.0 * dot_dn
    return dx - two_dot * nx, dy - two_dot * ny, dz - two_dot * nz


@njit(fastmath=True, cache=True)
def _scattering_angle(
    dx: float, dy: float, dz: float,
    sx: float, sy: float, sz: float,
) -> float:
    """
    计算入射方向与散射方向之间的散射角 Θ（弧度，范围 [0, π]）。

    Θ = arccos(k̂ᵢ · k̂ₛ)，Θ=0 为前向散射，Θ=π 为后向散射。

    :param dx, dy, dz: 入射方向（单位向量）
    :param sx, sy, sz: 散射方向（单位向量）

    :return: 散射角 Θ ∈ [0, π]（弧度）
    """
    cos_theta = dx * sx + dy * sy + dz * sz
    # 数值保护，防止超出 arccos 定义域 [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return np.arccos(cos_theta)


# ==============================================================================
#  Numba 并行核心（蒙特卡洛相矩阵光线追踪）
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _compute_phase_matrix_monte_carlo(
    medium: np.ndarray,
    n_rays: int,
    n_angle_bins: int,
    dir_epsilon: float,
    epsilon: float,
    max_segment_voxels: float,
    n_air: float,
    n_ice: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    蒙特卡洛相矩阵计算核心（Numba 并行加速）。

    对每条光线，用 DDA 找到第一个冰-空气界面，在该界面处应用菲涅尔理论，
    计算反射光和折射光的散射角与偏振权重，统计到各散射角区间的累加器中。

    只追踪到第一个界面（单次散射假设），与论文 Section II-C 描述一致。

    :param medium:            二值化三维介质 (True=冰, False=空气)，形状 (N, N, N)
    :param n_rays:            蒙特卡洛光线总数
    :param n_angle_bins:      散射角区间数，[0, π] 均匀划分
    :param dir_epsilon:       DDA 方向分量零保护
    :param epsilon:           界面偏移量（防止光线卡在界面）
    :param max_segment_voxels: DDA 最大追踪步长（体素数）
    :param n_air:             空气折射率
    :param n_ice:             冰折射率

    :return: (acc_P11, acc_P22, n_valid)
        acc_P11[n_angle_bins] : 各角度区间 s 偏振（P₁₁）权重累积（未归一化）
        acc_P22[n_angle_bins] : 各角度区间 p 偏振（P₂₂）权重累积（未归一化）
        n_valid               : 成功命中冰-空气界面的有效光线数
    """
    resolution = medium.shape[0]

    # 线程局部累加器（每条光线一行，避免并行写竞争）
    local_P11 = np.zeros((n_rays, n_angle_bins))
    local_P22 = np.zeros((n_rays, n_angle_bins))
    local_valid = np.zeros(n_rays, dtype=np.int64)

    for ray_idx in prange(n_rays):

        # ------------------------------------------------------------------
        # 1. 随机起点（允许在冰相或空气相中出发，与论文一致）
        # ------------------------------------------------------------------
        ox = np.random.uniform(1.0, resolution - 2.0)
        oy = np.random.uniform(1.0, resolution - 2.0)
        oz = np.random.uniform(1.0, resolution - 2.0)

        init_phase = get_voxel_phase(ox, oy, oz, medium, resolution)
        if init_phase < 0:
            continue  # 起点越界，丢弃

        # ------------------------------------------------------------------
        # 2. 随机入射方向
        # ------------------------------------------------------------------
        dx, dy, dz = random_unit_vector_no_trig()

        # ------------------------------------------------------------------
        # 3. DDA 算法追踪至第一个相界面
        # ------------------------------------------------------------------
        step_x = 1 if dx > 0.0 else -1
        step_y = 1 if dy > 0.0 else -1
        step_z = 1 if dz > 0.0 else -1

        # 每步跨过一个体素面所需的参数 t 增量
        delta_tx = abs(1.0 / (dx + dir_epsilon))
        delta_ty = abs(1.0 / (dy + dir_epsilon))
        delta_tz = abs(1.0 / (dz + dir_epsilon))

        # 到达当前体素各方向最近面的初始 t 值
        t_max_x = (np.floor(ox) + 1.0 - ox) * delta_tx if dx > 0.0 else (ox - np.floor(ox)) * delta_tx
        t_max_y = (np.floor(oy) + 1.0 - oy) * delta_ty if dy > 0.0 else (oy - np.floor(oy)) * delta_ty
        t_max_z = (np.floor(oz) + 1.0 - oz) * delta_tz if dz > 0.0 else (oz - np.floor(oz)) * delta_tz

        curr_ix = int(ox)
        curr_iy = int(oy)
        curr_iz = int(oz)
        seg_t = 0.0          # 当前已走的参数 t（体素单位）
        found_iface = False
        iface_t = 0.0        # 界面处的参数 t

        while seg_t < max_segment_voxels:
            # 选择最近的体素面并推进
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    seg_t = t_max_x
                    t_max_x += delta_tx
                    curr_ix += step_x
                else:
                    seg_t = t_max_z
                    t_max_z += delta_tz
                    curr_iz += step_z
            else:
                if t_max_y < t_max_z:
                    seg_t = t_max_y
                    t_max_y += delta_ty
                    curr_iy += step_y
                else:
                    seg_t = t_max_z
                    t_max_z += delta_tz
                    curr_iz += step_z

            # 边界检测（留 1 体素余量保证法线计算有效）
            if (curr_ix < 1 or curr_ix >= resolution - 1 or
                    curr_iy < 1 or curr_iy >= resolution - 1 or
                    curr_iz < 1 or curr_iz >= resolution - 1):
                break

            # 相界面检测：进入与起始相不同的体素
            new_phase = get_voxel_phase(curr_ix, curr_iy, curr_iz, medium, resolution)
            if new_phase != init_phase:
                found_iface = True
                iface_t = seg_t
                break

        if not found_iface:
            continue  # 未命中界面，丢弃

        # ------------------------------------------------------------------
        # 4. 计算界面交点坐标
        # ------------------------------------------------------------------
        hit_x = ox + dx * iface_t
        hit_y = oy + dy * iface_t
        hit_z = oz + dz * iface_t

        # ------------------------------------------------------------------
        # 5. 计算表面法线（中心差分梯度，初始指向空气侧）
        # ------------------------------------------------------------------
        nx_v, ny_v, nz_v = get_surface_normal(hit_x, hit_y, hit_z, medium)
        if nx_v == 0.0 and ny_v == 0.0 and nz_v == 0.0:
            continue  # 法线无法确定（平坦内部区域），丢弃

        # 确保法线指向入射侧（即 dot(k̂ᵢ, n̂) < 0）
        dot_dn = dx * nx_v + dy * ny_v + dz * nz_v
        if dot_dn > 0.0:
            nx_v, ny_v, nz_v = -nx_v, -ny_v, -nz_v
            dot_dn = -dot_dn

        cos_i = -dot_dn  # 入射角余弦（正值）
        if cos_i < 1e-6:
            continue  # 掠射角，法线不可靠，丢弃

        # ------------------------------------------------------------------
        # 6. 确定界面两侧折射率
        # ------------------------------------------------------------------
        n1 = n_ice if init_phase == 1 else n_air   # 入射介质
        n2 = n_air if init_phase == 1 else n_ice   # 透射介质

        # ------------------------------------------------------------------
        # 7. 偏振菲涅尔系数
        #    R_s → P₁₁ (s/ê₁)，R_p → P₂₂ (p/ê₂)
        # ------------------------------------------------------------------
        R_s, R_p, T_s, T_p, is_tir = _fresnel_polarized(cos_i, n1, n2)

        # ------------------------------------------------------------------
        # 8. 反射光散射角 Θᵣ：反射方向 k̂ᵣ = k̂ᵢ - 2(k̂ᵢ·n̂)n̂
        # ------------------------------------------------------------------
        rx, ry, rz = _reflect_direction(dx, dy, dz, nx_v, ny_v, nz_v, dot_dn)
        theta_r = _scattering_angle(dx, dy, dz, rx, ry, rz)

        bin_r = int(theta_r / np.pi * n_angle_bins)
        if bin_r >= n_angle_bins:
            bin_r = n_angle_bins - 1
        local_P11[ray_idx, bin_r] += R_s
        local_P22[ray_idx, bin_r] += R_p

        # ------------------------------------------------------------------
        # 9. 折射光散射角 Θₜ（全内反射时跳过）
        #    refract_vector 需要法线指向入射侧（dot_dn < 0），当前已满足
        # ------------------------------------------------------------------
        if not is_tir:
            tx, ty, tz = refract_vector(dx, dy, dz, nx_v, ny_v, nz_v, n1, n2)
            # refract_vector 返回 (0,0,0) 表示全反射（数值二次保护）
            if not (tx == 0.0 and ty == 0.0 and tz == 0.0):
                theta_t = _scattering_angle(dx, dy, dz, tx, ty, tz)
                bin_t = int(theta_t / np.pi * n_angle_bins)
                if bin_t >= n_angle_bins:
                    bin_t = n_angle_bins - 1
                local_P11[ray_idx, bin_t] += T_s
                local_P22[ray_idx, bin_t] += T_p

        local_valid[ray_idx] = 1

    # ------------------------------------------------------------------
    # 10. 汇总各线程局部结果
    # ------------------------------------------------------------------
    acc_P11 = np.zeros(n_angle_bins)
    acc_P22 = np.zeros(n_angle_bins)
    n_valid = 0
    for i in range(n_rays):
        n_valid += local_valid[i]
        for j in range(n_angle_bins):
            acc_P11[j] += local_P11[i, j]
            acc_P22[j] += local_P22[i, j]

    return acc_P11, acc_P22, n_valid


# ==============================================================================
#  上层计算类
# ==============================================================================

class PhaseMatrixCalculator:
    """
    相矩阵计算器

    在几何光学极限下，通过蒙特卡洛光线追踪计算双连续雪介质的散射相矩阵
    元素 P₁₁(Θ) 和 P₂₂(Θ)（1-2 参考系）。

    参考：Xiong et al., IEEE TGRS, 2015, Section II-C

    使用示例
    --------
    >>> calc = PhaseMatrixCalculator(snow_medium, kappa_s=500.0)
    >>> P11, P22, angles = calc.run_simulation(n_rays=100000)
    >>> calc.plot_results()
    >>> g = calc.get_asymmetry_parameter()
    """

    def __init__(self, medium_instance, kappa_s: float) -> None:
        """
        初始化相矩阵计算器。

        :param medium_instance: BicontinuousMedium 实例（需已调用 generate()）
        :param kappa_s: 散射系数 κ_s [m⁻¹]，由消光系数减去吸收系数得到：
                        κ_s = κ_e - κ_a
        """
        self.medium: np.ndarray = medium_instance.get_binary_medium()
        self.L_physical: float = medium_instance.L        # 介质物理尺寸 [m]
        self.resolution: int = medium_instance.resolution # 网格分辨率
        self.voxel_len: float = self.L_physical / self.resolution  # 体素边长 [m]
        self.kappa_s: float = kappa_s

        # 结果存储（run_simulation 后填充）
        self.P11: Optional[np.ndarray] = None   # [m⁻¹]
        self.P22: Optional[np.ndarray] = None   # [m⁻¹]
        self.angles: Optional[np.ndarray] = None  # [rad]
        self._n_angle_bins: int = None

    # ------------------------------------------------------------------
    # 主仿真接口
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        n_rays: int = 100000,
        n_angle_bins: int = 180,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        运行蒙特卡洛相矩阵仿真。

        :param n_rays:       蒙特卡洛光线数量（越大统计噪声越小，建议 ≥ 50000）
        :param n_angle_bins: 散射角区间数，将 [0°, 180°] 均匀划分

        :return: (P11, P22, angles)
            P11   : P₁₁ 相矩阵元素，形状 (n_angle_bins,)，单位 m⁻¹
            P22   : P₂₂ 相矩阵元素，形状 (n_angle_bins,)，单位 m⁻¹
            angles: 散射角中心值，形状 (n_angle_bins,)，单位 rad
            （若失败均返回 None）
        """
        self._n_angle_bins = n_angle_bins

        print("=" * 58)
        print("🚀 启动相矩阵蒙特卡洛仿真 (Phase Matrix Simulation)")
        print(f"   光线数:    {n_rays:,}")
        print(f"   角度区间:  {n_angle_bins}  （[0°, 180°]）")
        print(f"   散射系数:  κ_s = {self.kappa_s:.2f} m⁻¹")
        print(f"   介质分辨率:{self.resolution}³ 体素")
        print("=" * 58)

        # 运行 Numba 并行核心
        print("🔄 正在进行光线追踪（仅追踪至第一个界面）...")
        acc_P11, acc_P22, n_valid = _compute_phase_matrix_monte_carlo(
            medium=self.medium,
            n_rays=n_rays,
            n_angle_bins=n_angle_bins,
            dir_epsilon=_DIR_EPSILON,
            epsilon=_EPSILON,
            max_segment_voxels=_MAX_SEGMENT_VOXELS,
            n_air=N_AIR,
            n_ice=N_ICE,
        )
        print(f"✔️  追踪完成！有效命中界面：{n_valid:,} / {n_rays:,} 条")

        if n_valid == 0:
            print("❌ 无有效光线命中界面，请检查介质参数或增大 n_rays。")
            return None, None, None

        # ------------------------------------------------------------------
        # 归一化：论文式 (16)/(17)
        #
        # 由 κ_s = π Σᵢ (P11(Θᵢ)+P22(Θᵢ)) sin(Θᵢ) ΔΘ 得：
        #
        #   P11(Θᵢ) = acc_P11[i] · κ_s / normalization
        #   P22(Θᵢ) = acc_P22[i] · κ_s / normalization
        #
        # 其中 normalization = π · Σⱼ (acc_P11[j]+acc_P22[j]) · sin(Θⱼ) · ΔΘ
        # （这里 acc_P11/acc_P22 已是汇总值，n_valid 的因子在分子分母相消）
        # ------------------------------------------------------------------
        delta_angle = np.pi / n_angle_bins
        angles = np.linspace(delta_angle / 2.0, np.pi - delta_angle / 2.0, n_angle_bins)

        acc_total = acc_P11 + acc_P22
        normalization = np.pi * np.sum(acc_total * np.sin(angles) * delta_angle)

        if normalization < 1e-20:
            print("⚠️  归一化分母过小（相矩阵权重不足），请增大 n_rays。")
            return None, None, None

        scale = self.kappa_s / normalization
        P11 = acc_P11 * scale
        P22 = acc_P22 * scale

        self.P11 = P11
        self.P22 = P22
        self.angles = angles

        # ------------------------------------------------------------------
        # 自检：重建 κ_s 并计算误差
        # ------------------------------------------------------------------
        kappa_s_check = np.pi * np.sum((P11 + P22) * np.sin(angles) * delta_angle)
        rel_err = abs(kappa_s_check - self.kappa_s) / self.kappa_s * 100.0

        print("=" * 58)
        print("✅ 相矩阵计算完成！")
        print(f"   κ_s 输入值:  {self.kappa_s:.4f} m⁻¹")
        print(f"   κ_s 重建值:  {kappa_s_check:.4f} m⁻¹")
        print(f"   重建相对误差: {rel_err:.2f}%  （理论上应为 0%）")
        print(f"   P₁₁ 峰值角:  {np.degrees(angles[np.argmax(P11)]):.1f}°")
        print(f"   P₂₂ 峰值角:  {np.degrees(angles[np.argmax(P22)]):.1f}°")
        print("=" * 58)

        return P11, P22, angles

    # ------------------------------------------------------------------
    # 可视化接口
    # ------------------------------------------------------------------

    def plot_results(self, wavelength_nm: Optional[float] = None) -> None:
        """
        绘制极坐标形式的相矩阵元素图，复现论文 Fig. 6 的风格。

        :param wavelength_nm: 波长 [nm]，用于图标题标注（可选）
        """
        if self.P11 is None:
            print("❌ 请先调用 run_simulation()。")
            return

        wl_str = f"λ = {wavelength_nm:.0f} nm, " if wavelength_nm else ""
        title = (f"Phase Matrix Elements in 1-2 Reference Frame\n"
                 f"({wl_str}κ_s = {self.kappa_s:.1f} m⁻¹, "
                 f"n_bins = {self._n_angle_bins})")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                                 subplot_kw={"projection": "polar"})
        fig.suptitle(title, fontsize=12)

        for ax, data, label in zip(
            axes,
            [self.P11, self.P22],
            ["$P_{11}$ (m$^{-1}$)", "$P_{22}$ (m$^{-1}$)"],
        ):
            ax.plot(self.angles, data, linewidth=1.8)
            ax.fill(self.angles, data, alpha=0.18)
            ax.set_theta_zero_location("N")   # 0° 在顶部（前向散射）
            ax.set_theta_direction(-1)        # 顺时针增大
            ax.set_title(label, fontsize=11, pad=18)
            # 添加 0°/180° 标注
            ax.set_thetagrids(
                range(0, 361, 30),
                labels=[f"{a}°" for a in range(0, 361, 30)],
                fontsize=7,
            )

        plt.tight_layout()
        os.makedirs("Results", exist_ok=True)
        out = f"Results/phase_matrix_polar_ks{self.kappa_s:.0f}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"📊 极坐标图已保存：{out}")
        plt.show()

    def plot_cartesian(self, wavelength_nm: Optional[float] = None) -> None:
        """
        绘制笛卡尔坐标（散射角 - 相矩阵元素）的半对数图。

        :param wavelength_nm: 波长 [nm]，用于图标题标注（可选）
        """
        if self.P11 is None:
            print("❌ 请先调用 run_simulation()。")
            return

        wl_str = f"λ = {wavelength_nm:.0f} nm" if wavelength_nm else "相矩阵"
        angles_deg = np.degrees(self.angles)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Phase Matrix Elements — {wl_str}  (κ_s = {self.kappa_s:.1f} m⁻¹)",
            fontsize=12,
        )

        for ax, data, label in zip(
            axes,
            [self.P11, self.P22],
            ["$P_{11}$ (m$^{-1}$)", "$P_{22}$ (m$^{-1}$)"],
        ):
            ax.semilogy(angles_deg, np.maximum(data, 1e-12), linewidth=1.5)
            ax.set_xlabel("Scattering Angle Θ (°)", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_xlim(0, 180)
            ax.set_xticks(range(0, 181, 30))
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        os.makedirs("Results", exist_ok=True)
        out = f"Results/phase_matrix_cartesian_ks{self.kappa_s:.0f}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"📊 笛卡尔图已保存：{out}")
        plt.show()

    # ------------------------------------------------------------------
    # 物理量导出
    # ------------------------------------------------------------------

    def get_asymmetry_parameter(self) -> float:
        """
        计算散射不对称因子 g（asymmetry parameter）。

        g = ∫ P₁₁(Θ) cos(Θ) sin(Θ) dΘ / ∫ P₁₁(Θ) sin(Θ) dΘ

        g ≈ 1 表示强前向散射，g ≈ 0 表示各向同性，g ≈ -1 表示后向散射。
        冰颗粒可见光波段典型值约为 0.87（文献参考值）。

        :return: g ∈ [-1, 1]
        """
        if self.P11 is None:
            print("❌ 请先调用 run_simulation()。")
            return 0.0

        delta_angle = np.pi / self._n_angle_bins
        sin_w = np.sin(self.angles) * delta_angle
        numerator = np.sum(self.P11 * np.cos(self.angles) * sin_w)
        denominator = np.sum(self.P11 * sin_w)
        g = float(numerator / denominator) if denominator > 1e-20 else 0.0

        print(f"📐 不对称因子 g = {g:.4f}  （冰颗粒典型值 ~0.87）")
        return g

    def get_phase_function(self) -> Optional[np.ndarray]:
        """
        返回归一化相函数 p(Θ) = (P₁₁(Θ) + P₂₂(Θ)) / (κ_s / 2π)。

        归一化条件：(1/4π) ∫ p(Θ) dΩ = 1。

        :return: 归一化相函数数组，形状 (n_angle_bins,)；失败返回 None
        """
        if self.P11 is None:
            return None
        p_unnorm = self.P11 + self.P22
        return p_unnorm / (self.kappa_s / (2.0 * np.pi))
