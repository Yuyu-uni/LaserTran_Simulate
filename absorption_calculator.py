import numpy as np
from numba import njit, prange, config, get_num_threads, set_num_threads
from basic_utils import (
    get_voxel_phase,
    random_unit_vector_no_trig,
    normalize_vector,
    get_surface_normal,
    fresnel_reflectivity,
    refract_vector,
    compute_ice_absorption,
    N_AIR,
    N_ICE,
)

# ==========================================
# 命名常量
# ==========================================
NUM_BINS: int = 200              # 距离分箱数量
EPSILON: float = 1e-5            # 浮点偏移量，避免边界卡死
DIR_EPSILON: float = 1e-20       # 方向分量零保护
MAX_SEGMENT_VOXELS: float = 50.0 # 单段最大步长（体素单位）
RENORM_INTERVAL: int = 10        # 方向向量归一化间隔（反射/折射次数）
PROGRESS_INTERVAL: int = 10      # 进度输出间隔


# ==========================================
# Numba 并行设置
# ==========================================
def print_numba_info():
    """打印 Numba 配置信息（用于调试）"""
    print("=" * 50)
    print("🔧 Numba 配置信息:")
    print(f"   并行线程数: {get_num_threads()}")
    print(f"   并行启用: {config.NUMBA_DEFAULT_NUM_THREADS}")
    print(f"   FastMath: 启用")
    print("=" * 50)


def configure_numba(num_threads=None):
    """
    配置 Numba 并行参数
    
    :param num_threads: 并行线程数，None 表示使用所有可用核心
    """
    if num_threads is not None:
        set_num_threads(num_threads)
        print(f"✅ Numba 并行线程数已设置为: {num_threads}")
    else:
        print(f"✅ Numba 使用默认线程数: {get_num_threads()}")


# ==========================================
# 1. Numba 加速内核 (Ray Tracing Core)
# ==========================================

@njit(parallel=True, fastmath=True, cache=True)
def _compute_absorption_dda_monte_carlo(
    medium, 
    ice_absorption, 
    voxel_len, 
    num_rays, 
    max_dist_m,
    num_bins,
    epsilon,
    dir_epsilon,
    max_segment_voxels,
    renorm_interval,
    n_air,
    n_ice,
):
    """
    统计 P_abs(L) = 1 - exp(-k * L_ice)
    使用 DDA 算法进行蒙特卡洛路径追踪，计算在双连续介质中传播的光线的吸收概率分布。
    
    :param medium: 二值化的三维介质 (numpy 3D array)
    :param ice_absorption: 冰的吸收系数 (m^-1)
    :param voxel_len: 介质单个体素边长 (m)
    :param num_rays: 发射的光线数量
    :param max_dist_m: 最大传播距离 (m)
    :param num_bins: 距离分箱数量
    :param epsilon: 浮点偏移量
    :param dir_epsilon: 方向分量零保护
    :param max_segment_voxels: 单段最大步长（体素单位）
    :param renorm_interval: 方向向量归一化间隔
    :param n_air: 空气折射率
    :param n_ice: 冰折射率
    
    :return: absorbed_accumulator: 每个距离节点的吸收概率累计值
    :return: ray_counts: 每个距离节点的光线存活数量
    """
    medium_bound = medium.shape[0]  # 假设介质为正方体
    bin_width = max_dist_m / num_bins
    
    # 使用线程局部累加器避免并行竞争条件
    num_threads = num_rays  # 每条光线一个局部数组
    local_absorbed = np.zeros((num_rays, num_bins))
    local_counts = np.zeros((num_rays, num_bins))

    for i in prange(num_rays):
        # 选取随机起点，确保在空气中
        while True:
            ox = np.random.uniform(1, medium_bound - 2)
            oy = np.random.uniform(1, medium_bound - 2)
            oz = np.random.uniform(1, medium_bound - 2)
            # 确保光线起点在空气中
            if get_voxel_phase(ox, oy, oz, medium, medium_bound) == 0:
                break
        
        dx, dy, dz = random_unit_vector_no_trig()
        curr_x, curr_y, curr_z = ox, oy, oz
        
        # 路径追踪变量
        L_total_m = 0.0   # 总路径长度
        L_ice = 0.0       # 在冰中的路径长度
        in_ice = False    # 当前是否在冰中(初始在空气中)
        bounce_count = 0  # 反射/折射计数，用于方向向量归一化
        
        while L_total_m < max_dist_m:
            # DDA 初始化
            ix, iy, iz = int(curr_x), int(curr_y), int(curr_z)

            step_x = 1 if dx > 0 else -1
            step_y = 1 if dy > 0 else -1
            step_z = 1 if dz > 0 else -1

            delta_x = abs(1.0 / (dx + dir_epsilon))
            delta_y = abs(1.0 / (dy + dir_epsilon))
            delta_z = abs(1.0 / (dz + dir_epsilon))

            dist_x = (np.floor(curr_x) + 1.0 - curr_x) * delta_x if dx > 0 else (curr_x - np.floor(curr_x)) * delta_x
            dist_y = (np.floor(curr_y) + 1.0 - curr_y) * delta_y if dy > 0 else (curr_y - np.floor(curr_y)) * delta_y
            dist_z = (np.floor(curr_z) + 1.0 - curr_z) * delta_z if dz > 0 else (curr_z - np.floor(curr_z)) * delta_z

            t_max_x, t_max_y, t_max_z = dist_x, dist_y, dist_z
            
            hit_interface = False
            segment_dist_vox = 0.0
            
            # 限制单段最大步长
            while segment_dist_vox < max_segment_voxels:
                # 寻找最近边界
                if t_max_x < t_max_y:
                    if t_max_x < t_max_z:
                        d_step = t_max_x - segment_dist_vox
                        segment_dist_vox = t_max_x
                        t_max_x += delta_x
                        ix += step_x
                    else:
                        d_step = t_max_z - segment_dist_vox
                        segment_dist_vox = t_max_z
                        t_max_z += delta_z
                        iz += step_z
                else:
                    if t_max_y < t_max_z:
                        d_step = t_max_y - segment_dist_vox
                        segment_dist_vox = t_max_y
                        t_max_y += delta_y
                        iy += step_y
                    else:
                        d_step = t_max_z - segment_dist_vox
                        segment_dist_vox = t_max_z
                        t_max_z += delta_z
                        iz += step_z
                
                # 已进入新位置，开始检测
                d_step_m = d_step * voxel_len  # 转换为米
                L_total_m += d_step_m
                if in_ice:
                    L_ice += d_step_m
                
                # 统计到对应距离桶中（使用线程局部累加器避免竞争条件）
                bin_idx = int(L_total_m / bin_width)
                if bin_idx < num_bins:
                    # 计算吸收概率: P_abs = 1 - exp(-k * L_ice)
                    p_abs = 1.0 - np.exp(-ice_absorption * L_ice)
                    local_absorbed[i, bin_idx] += p_abs
                    local_counts[i, bin_idx] += 1.0

                # 越界检测
                if (ix < 1 or ix >= medium_bound - 1 or 
                    iy < 1 or iy >= medium_bound - 1 or 
                    iz < 1 or iz >= medium_bound - 1):
                    L_total_m = max_dist_m + 1.0
                    break
                
                # 处理边界碰撞
                target_phase = 1 if in_ice else 0
                if get_voxel_phase(ix, iy, iz, medium, medium_bound) != target_phase:
                    hit_interface = True
                    break
                
            # 单只光线 一段直线段处理完毕
            
            # 已超出最大检测距离
            if L_total_m > max_dist_m:
                break
            if not hit_interface:
                break
            
            # 处理界面反射/折射
            hit_x = curr_x + dx * segment_dist_vox
            hit_y = curr_y + dy * segment_dist_vox
            hit_z = curr_z + dz * segment_dist_vox
            
            nx_vec, ny_vec, nz_vec = get_surface_normal(hit_x, hit_y, hit_z, medium)
            if nx_vec == 0 and ny_vec == 0 and nz_vec == 0:
                break
            dot_val = dx * nx_vec + dy * ny_vec + dz * nz_vec
            
            if dot_val > 0:
                nx_vec, ny_vec, nz_vec = -nx_vec, -ny_vec, -nz_vec
                dot_val = -dot_val
                
            # 计算菲涅尔反射率
            # 确定折射率
            n1 = n_ice if in_ice else n_air
            n2 = n_air if in_ice else n_ice
            R = fresnel_reflectivity(n1, n2, -dot_val)
            
            if np.random.random() < R:
                # 反射
                dx = dx - 2 * dot_val * nx_vec
                dy = dy - 2 * dot_val * ny_vec
                dz = dz - 2 * dot_val * nz_vec
                
                # 方向向量再归一化（定期执行以消除浮点累积误差）
                bounce_count += 1
                if bounce_count % renorm_interval == 0:
                    dx, dy, dz = normalize_vector(dx, dy, dz)
                
                curr_x = hit_x + dx * epsilon
                curr_y = hit_y + dy * epsilon
                curr_z = hit_z + dz * epsilon
            else:
                # 折射
                tx, ty, tz = refract_vector(dx, dy, dz, nx_vec, ny_vec, nz_vec, n1, n2)
                if tx == 0 and ty == 0 and tz == 0:
                    # 全反射
                    dx = dx - 2 * dot_val * nx_vec
                    dy = dy - 2 * dot_val * ny_vec
                    dz = dz - 2 * dot_val * nz_vec
                    
                    bounce_count += 1
                    if bounce_count % renorm_interval == 0:
                        dx, dy, dz = normalize_vector(dx, dy, dz)
                    
                    curr_x = hit_x + dx * epsilon
                    curr_y = hit_y + dy * epsilon
                    curr_z = hit_z + dz * epsilon
                else:
                    dx, dy, dz = tx, ty, tz
                    
                    # 方向向量再归一化
                    bounce_count += 1
                    if bounce_count % renorm_interval == 0:
                        dx, dy, dz = normalize_vector(dx, dy, dz)
                    
                    curr_x = hit_x + dx * epsilon
                    curr_y = hit_y + dy * epsilon
                    curr_z = hit_z + dz * epsilon
                    in_ice = not in_ice
    
    # 汇总线程局部累加器到全局结果（修复并行竞争条件）
    absorbed_accumulator = np.zeros(num_bins)
    ray_counts = np.zeros(num_bins)
    for ray_idx in range(num_rays):
        for bin_idx in range(num_bins):
            absorbed_accumulator[bin_idx] += local_absorbed[ray_idx, bin_idx]
            ray_counts[bin_idx] += local_counts[ray_idx, bin_idx]
    
    # 计算平均吸收概率
    avg_absorption = np.zeros(num_bins)
    for bin_idx in range(num_bins):
        if ray_counts[bin_idx] > 0:
            avg_absorption[bin_idx] = absorbed_accumulator[bin_idx] / ray_counts[bin_idx]
    
    # 生成距离数组，每个bin的中心位置
    distance_array = np.linspace(bin_width / 2, max_dist_m - bin_width / 2, num_bins)
    
    # 组合成 (N, 2) 数组: 第一列为距离L，第二列为平均吸收概率
    absorption_results_list = np.column_stack((distance_array, avg_absorption))
    
    return absorption_results_list, ray_counts


# ==========================================
# 2. 上层计算类 (Calculator Class)
# ==========================================
class AbsorptionCalculator:
    """吸收系数计算器"""
    
    def __init__(self, medium_instance, extinction_coefficient):
        """
        初始化吸收计算器。
        
        :param medium_instance: BicontinuousMedium 实例
        :param extinction_coefficient: 消光系数 (m^-1)
        """
        self.medium = medium_instance.get_binary_medium()
        self.L_physical = medium_instance.L
        self.resolution = medium_instance.resolution

        self.voxel_len = self.L_physical / self.resolution  # 体素边长 (m)

        self.fv = medium_instance.fv
        self.extinction_coefficient = extinction_coefficient
        
        self.absorption_results = None  # 储存吸收结果数组
        self.absorption_coefficient = None  # 储存计算得到的吸收系数 (m^-1)

    def run_simulation(self, wavelength_nm, num_rays=50000, max_dist_mm=50.0, num_threads=None):
        """
        运行吸收计算模拟。
        
        :param wavelength_nm: 激光波长 (nm)
        :param num_rays: 发射的光线数量
        :param max_dist_mm: 最大传播距离 (mm)
        :param num_threads: 并行线程数，默认None使用所有CPU核心
        
        :return: absorption_coefficient: 拟合得到的吸收系数
        """
        # 0. 配置并行参数
        print_numba_info()
        configure_numba(num_threads)
        
        ice_absorption = compute_ice_absorption(wavelength_nm)  # m^-1
        max_dist_m = max_dist_mm * 1e-3  # 转换为米

        print("=" * 50)
        print("🚀 启动吸收计算模拟...")
        print(f"🌟 波长: {wavelength_nm} nm | 冰吸收系数: {ice_absorption:.4f} m^-1")
        print(f"📊 追踪光线数量: {num_rays} | 最大统计距离: {max_dist_mm} mm")
        print(f"📦 距离分箱数量: {NUM_BINS} | 方向归一化间隔: {RENORM_INTERVAL}")
        print("=" * 50)
        
        # 运行蒙特卡洛吸收计算
        print("🔄 正在进行路径追踪计算...")
        self.absorption_results, ray_counts = _compute_absorption_dda_monte_carlo(
            medium=self.medium,
            ice_absorption=ice_absorption,
            voxel_len=self.voxel_len,
            num_rays=num_rays,
            max_dist_m=max_dist_m,
            num_bins=NUM_BINS,
            epsilon=EPSILON,
            dir_epsilon=DIR_EPSILON,
            max_segment_voxels=MAX_SEGMENT_VOXELS,
            renorm_interval=RENORM_INTERVAL,
            n_air=N_AIR,
            n_ice=N_ICE,
        )
        
        # 输出统计信息
        total_valid = np.sum(ray_counts)
        avg_valid = np.mean(ray_counts)
        print(f"✔️ 路径追踪完成！总有效采样点: {total_valid:.0f}，平均每箱: {avg_valid:.1f}")
        
        # 曲线拟合，提取吸收系数
        self.absorption_coefficient = self._absorption_curve_fit(ice_absorption)
        
        return self.absorption_coefficient

    def _absorption_curve_fit(self, ice_absorption):
        """
        进行吸收概率曲线最小二乘拟合，以提取吸收系数。
        
        :param ice_absorption: 冰的吸收系数
        
        :return: absorption_coefficient: 拟合得到的吸收系数
        """
        # 理想吸收概率模型
        def ideal_absorption_model(L, absorption_coefficient):
            return (1.0 - np.exp(-absorption_coefficient * L))
        
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("❌ 未安装 scipy 库，无法进行曲线拟合。请安装 scipy 后重试。")
            return None
        
        print("🔍 开始进行曲线拟合，提取吸收系数...")
        
        # 数据截断
        L_min_m = 5e-3   
        L_max_m = 20.0e-3 
        
        L_data = self.absorption_results[:, 0]
        P_data = self.absorption_results[:, 1]
        
        # 创建截断掩码
        mask = (L_data >= L_min_m) & (L_data <= L_max_m)
        L_fit = L_data[mask]
        P_fit = P_data[mask]
        
        print(f"📐 数据截断范围: [{L_min_m*1000:.1f}, {L_max_m*1000:.1f}] mm")
        print(f"📊 用于拟合的数据点数: {len(L_fit)} / {len(L_data)}")
        
        try:
            popt, pcov = curve_fit(
                ideal_absorption_model,
                L_fit,  # 截断后的 L
                P_fit,  # 截断后的 P_abs
                p0=[ice_absorption * self.fv],  # 初始猜测值
                bounds=(0, np.inf)
            )
            absorption_coefficient = popt[0]
            print("=" * 50)
            print(f"🎉 曲线拟合成功！拟合得到的吸收系数为: {absorption_coefficient:.3f} m^-1")
            print("=" * 50)
            return absorption_coefficient
        except Exception as e:
            print(f"❌ 曲线拟合失败：{e}")
            return 0.0
   
    def plot_results(self):
        """
        绘制吸收概率计算结果及拟合曲线。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("❌ 未安装 matplotlib 库，无法绘制结果。请安装 matplotlib 后重试。")
            return
        
        if self.absorption_results is None:
            print("❌ 尚未进行吸收概率计算，请先运行 run_simulation 方法。")
            return
        
        import os
        
        plt.figure(figsize=(8, 6))
        # 绘制吸收概率曲线 (0 -> 1)
        plt.plot(
            self.absorption_results[:, 0] * 1000, 
            self.absorption_results[:, 1], 
            'k.', 
            markersize=2, 
            label='Simulated $P_{abs}$'
        )
        
        # 绘制拟合线
        y_fit = (1.0 - np.exp(-self.absorption_coefficient * self.absorption_results[:, 0]))
        plt.plot(
            self.absorption_results[:, 0] * 1000, 
            y_fit, 
            'r-', 
            linewidth=2, 
            label=f'Fit: $(1 - e^{{-{self.absorption_coefficient:.1f} L}})$'
        )
        
        plt.xlabel('Path Length $L_{total}$ (mm)')
        plt.ylabel('Probability of Absorption $P_a(L)$')
        plt.title('Absorption Coefficient Extraction---$\\kappa_a = {:.1f}$ m$^{{-1}}$'.format(self.absorption_coefficient))
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # 保存图片到 Results 目录
        output_path = f"Results/absorption_coefficient_{self.absorption_coefficient:.0f}.png"
        os.makedirs("Results", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"📊 结果图已保存至: {output_path}")
        plt.show()