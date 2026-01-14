import numpy as np
import os
from numba import njit, prange, config, get_num_threads, set_num_threads
from basic_utils import compute_ice_absorption, get_voxel_phase, random_unit_vector_no_trig

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

# @njit(fastmath=True, cache=True)
# def _get_voxel_phase(x, y, z, medium, nx):
#     '''
#     获取指定坐标处的介质相。
#     坐标以体素为单位  
    
#     :param x: 指定坐标位置x
#     :param y: 指定坐标位置y
#     :param z: 指定坐标位置z
#     :param medium: 二值化的三维介质
#     :param nx: 介质正方体边长

#     :return: 1(Ice), 0(Air), -1(超出介质范围)
#     '''
#     # 连续坐标转换为体素
#     ix = int(x)
#     iy = int(y)
#     iz = int(z)
    
#     if (ix < 0 or ix >= nx or iy < 0 or iy >= nx or iz < 0 or iz >= nx):
#         return -1  # 超出介质范围
    
#     return 1 if medium[ix, iy, iz] else 0

# @njit(fastmath=True, cache=True)
# def random_unit_vector_no_trig():
#     """
#     生成球面均匀分布的单位向量，不使用三角函数。
#     原理：在立方体内随机取点，如果在球内则归一化，否则重取。
#     对于 Numba 来说，这比计算 sin/cos 快得多。
#     """
#     while True:
#         # 在 [-1, 1] 区间内生成 x, y, z
#         x = np.random.uniform(-1.0, 1.0)
#         y = np.random.uniform(-1.0, 1.0)
#         z = np.random.uniform(-1.0, 1.0)
#         length_sq = x*x + y*y + z*z
#         # 如果点在单位球内（且不是原点）
#         if 0.0001 < length_sq <= 1.0:
#             inv_len = 1.0 / np.sqrt(length_sq)
#             return x * inv_len, y * inv_len, z * inv_len

@njit(fastmath=True, parallel=True, cache=True)
def _compute_poe_classic_monte_carlo(medium, ice_absorption, L_in_voxel_units, rays_num, step_size = 0.5):
    '''
    传统蒙特卡洛方法，路径追踪计算消光概率。
    
    :param medium: 二值化的三维介质
    :param ice_absorption: 冰的吸收系数
    :param L_in_voxel_units: Li 光线目标长度 以体素单位表示
    :param rays_num: 发射光线数量
    :param step_size: 步长，每多少个体素单位进行一次检测
    
    :return avrg_POE: 长度Li下的平均消光概率
    :return valid_rays: 未射出边界的有效光线数量
    '''
    medium_bound = medium.shape[0]  # 假设介质为正方体
    sum_POE = 0.
    valid_rays_counter = 0
    
    # 并行循环发射光线
    for i in prange(rays_num):
        # Step1.选取随机起点
        start_x = np.random.uniform(0.0, float(medium_bound))
        start_y = np.random.uniform(0.0, float(medium_bound))
        start_z = np.random.uniform(0.0, float(medium_bound))
        # 获取起点体素的相
        start_phase = get_voxel_phase(start_x, start_y, start_z, medium, medium_bound)

        # Step2.向随机方向发射射线
        # phi = np.random.uniform(0, 2 * np.pi)
        # cos_theta = np.random.uniform(-1, 1)
        # sin_theta = np.sqrt(1 - cos_theta ** 2)
        
        # dir_x = sin_theta * np.cos(phi)
        # dir_y = sin_theta * np.sin(phi)
        # dir_z = cos_theta
        dir_x, dir_y, dir_z = random_unit_vector_no_trig()
        # Step3.路径追踪
        curr_dist = 0. # 当前路径长度
        hit_boundary = False
        out_of_bound = False
        curr_phase = start_phase
        
        # 从起点开始向终点追踪，每步长检测是否出界或遇相变
        while curr_dist < L_in_voxel_units:
            curr_dist += step_size
            # 修正最后一步
            if curr_dist > L_in_voxel_units:
                curr_dist = L_in_voxel_units

            curr_x = start_x + dir_x * curr_dist
            curr_y = start_y + dir_y * curr_dist
            curr_z = start_z + dir_z * curr_dist
            
            curr_phase = get_voxel_phase(curr_x, curr_y, curr_z, medium, medium_bound)

            if curr_phase == -1:
                out_of_bound = True
                break
            if curr_phase != start_phase:
                hit_boundary = True
                break
            
        # Step4.判断结果，计算消光概率
        if out_of_bound:
            continue  # 光线射出边界，丢弃该光线
        
        valid_rays_counter += 1  # 有效光线计数
        if hit_boundary:
            sum_POE += 1.0  # 碰到相变，消光
        else:
            # 未碰到相变，根据吸收系数计算存活概率
            if start_phase == 1:  # Ice
                sum_POE += 1.0 - np.exp(-ice_absorption * L_in_voxel_units)
            else:  # Air
                sum_POE += 0.0  # Air 不吸收
    # 所有光线追踪结束 计算平均消光概率      
    if valid_rays_counter == 0:
        avrg_POE = 0.0
        valid_rays = 0
            
    else:
        avrg_POE = sum_POE / valid_rays_counter
        valid_rays = valid_rays_counter
            
    return avrg_POE, valid_rays

      
@njit(fastmath=True, parallel=True, cache=True)            
def _compute_poe_dda_monte_carlo(medium, ice_absorption, L_in_voxel_units, rays_num):
    '''
    DDA算法蒙特卡洛方法，路径追踪计算消光概率。
    速度更快，解决穿墙误差问题。
    NOTE: 使用拒绝采样法选取起点，确保起点在介质内，避免三角运算减缓速度。
        
    :param medium: 二值化的三维介质
    :param ice_absorption: 冰的吸收系数
    :param L_in_voxel_units: Li 光线目标长度 以体素单位表示
    :param rays_num: 发射光线数量
    
    :return avrg_POE: 长度Li下的平均消光概率
    :return valid_rays: 未射出边界的有效光线数量
    '''   
    medium_bound = medium.shape[0]  # 假设介质为正方体
    sum_POE = 0.
    valid_rays_counter = 0    
    
    # 并行循环发射光线
    for i in prange(rays_num):
        # Step1.选取随机起点，确保起点在介质内
        start_x = np.random.uniform(0.0, float(medium_bound))
        start_y = np.random.uniform(0.0, float(medium_bound))
        start_z = np.random.uniform(0.0, float(medium_bound))
        
        # 将随机生成的浮点数起点离散化 转换为整数体素索引
        index_x = int(start_x)
        index_y = int(start_y)
        index_z = int(start_z)
        
        # 边界保护
        if (index_x >= medium_bound): index_x = medium_bound - 1
        if (index_y >= medium_bound): index_y = medium_bound - 1
        if (index_z >= medium_bound): index_z = medium_bound - 1     
    
        start_phase = get_voxel_phase(index_x, index_y, index_z, medium, medium_bound)
        # Step2.初始化随机方向发射射线
        dir_x, dir_y, dir_z = random_unit_vector_no_trig()
        
        # DDA算法初始化
        # 避免除零
        if abs(dir_x) <= 1.0e-10: dir_x = 1.0e-10
        if abs(dir_y) <= 1.0e-10: dir_y = 1.0e-10
        if abs(dir_z) <= 1.0e-10: dir_z = 1.0e-10
        
        # 计算步进方向和xyz方向网格间距delta
        step_x = 1 if dir_x > 0 else -1
        step_y = 1 if dir_y > 0 else -1
        step_z = 1 if dir_z > 0 else -1
        
        t_delta_x = abs(1.0 / dir_x)
        t_delta_y = abs(1.0 / dir_y)
        t_delta_z = abs(1.0 / dir_z)
        
        # 计算初始t_max 这是累计从起点到下一个边界的距离
        distance_to_next_x = (index_x + 1 - start_x) if step_x > 0 else (start_x - index_x)
        distance_to_next_y = (index_y + 1 - start_y) if step_y > 0 else (start_y - index_y)
        distance_to_next_z = (index_z + 1 - start_z) if step_z > 0 else (start_z - index_z)

        t_max_x = distance_to_next_x * t_delta_x
        t_max_y = distance_to_next_y * t_delta_y
        t_max_z = distance_to_next_z * t_delta_z
        
        travelled_distance = 0.0
        hit_boundary = False
        out_of_bound = False
        
        # ---------------------------------------------------------
        # 核心 DDA 循环 
        # ---------------------------------------------------------        
        while travelled_distance < L_in_voxel_units:
            # 找到下一个最近的边界，并更新位置
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    # 下一个边界是 x 方向
                    index_x += step_x
                    if index_x < 0 or index_x >= medium_bound:
                        out_of_bound = True
                        break
                    travelled_distance = t_max_x
                    t_max_x += t_delta_x # 增加到下一个x边界的距离
                else:
                    # 下一个边界是 z 方向
                    index_z += step_z
                    if index_z < 0 or index_z >= medium_bound:
                        out_of_bound = True
                        break
                    travelled_distance = t_max_z
                    t_max_z += t_delta_z # 增加到下一个z边界的距离
            else:# t_max_y <= t_max_x
                if t_max_y < t_max_z:
                    # 下一个边界是 y 方向
                    index_y += step_y
                    if index_y < 0 or index_y >= medium_bound:
                        out_of_bound = True
                        break
                    travelled_distance = t_max_y
                    t_max_y += t_delta_y # 增加到下一个y边界的距离
                else:
                    # 下一个边界是 z 方向
                    index_z += step_z
                    if index_z < 0 or index_z >= medium_bound:
                        out_of_bound = True
                        break
                    travelled_distance = t_max_z
                    t_max_z += t_delta_z # 增加到下一个z边界的距离

            # 此时已到达新的体素，检查相变（碰撞）
            if travelled_distance <= L_in_voxel_units:
                curr_phase = get_voxel_phase(index_x, index_y, index_z, medium, medium_bound)
                if curr_phase != start_phase:
                    hit_boundary = True
                    break # 碰到相变，退出循环
        # 单只光线追踪结束 统计结果
        if out_of_bound:
            continue  # 光线射出边界，丢弃该光线
        
        valid_rays_counter += 1  # 有效光线计数
        if hit_boundary:
            sum_POE += 1.0  # 碰到相变，消光            
        elif start_phase == 1:  # Ice
            # 未碰到相变，根据吸收系数计算存活概率
            sum_POE += 1.0 - np.exp(-ice_absorption * L_in_voxel_units)
    # 所有光线追踪结束 计算平均消光概率
    if valid_rays_counter == 0:
        avrg_POE = 0.0
        valid_rays = 0
    else:
        avrg_POE = sum_POE / valid_rays_counter
        valid_rays = valid_rays_counter
    return avrg_POE, valid_rays    
    
            
# ==========================================
# 2. 上层计算类 (Calculator Class)
# ==========================================
class ExtinctionCalculator:
    def __init__(self, medium_instance):
        '''
        消光计算器初始化。
        
        :param medium_instance: BicontinuousMedium 实例
        '''
        
        self.binary_medium = medium_instance.get_binary_medium()
        self.L_in_physical_units = medium_instance.L  # 介质物理边长 (m)
        self.resolution = medium_instance.resolution  # 介质分辨率 (体素数)
        self.voxel_size = self.L_in_physical_units / self.resolution  # 体素大小 (m/体素)

        self.poe_results_list = None  # 存储消光概率计算结果
        self.extinction_coefficient = None  # 存储计算得到的消光系数
        

    def run_simulation(self, wavelength_nm, rays_num = 50000, lengths_num = None, num_threads = None):
        '''
        执行消光系数计算流程，默认计算从单个体素到整个介质边长范围内的消光概率。
        
        :param wavelength_nm: 波长 (nm)
        :param rays_num: 发射光线数量
        :param lengths_num: 控制选取多少个Li，默认None表示计算从1个体素到介质边长范围内的所有整数路径长度。
        :param num_threads: 并行线程数，默认None使用所有CPU核心
        '''
        # 0. 配置并行参数
        print_numba_info()
        configure_numba(num_threads)
        
        # 1. 计算冰的吸收系数
        ice_absorption = compute_ice_absorption(wavelength_nm)
        # NOTE: 将吸收系数转换为体素单位
        ice_absorption_voxel_units = ice_absorption * self.voxel_size
        
        print("=" * 50)
        print(f"🌟计算波长 {wavelength_nm} nm 下的消光概率，发射光线数量：{rays_num}，冰的吸收系数：{ice_absorption:.3f} m^-1 ...")
        print("=" * 50)
        
        # 2. 确定路径长度Li列表
        # 从1个体素到介质边长范围内的所有整数路径长度
        max_length_in_voxel_units = self.resolution
        if lengths_num is None:
            lengths_list = np.arange(1, max_length_in_voxel_units + 1)
        else:
            lengths_list = np.linspace(1, max_length_in_voxel_units, lengths_num).astype(int)
            lengths_list = np.unique(lengths_list)  # 去重

        poe_results_list = []
        
        # 3. 循环计算各个路径长度下的消光概率 默认循环{self.resolution}次
        print(f"🚀开始路径追踪计算，共计 {len(lengths_list)} 个路径长度...")
        for i, L_in_voxel_units in enumerate(lengths_list):
            avrg_POE, valid_rays = _compute_poe_dda_monte_carlo(
                medium=self.binary_medium,
                ice_absorption=ice_absorption_voxel_units,
                L_in_voxel_units=L_in_voxel_units,
                rays_num=rays_num
            )
            # 将结果存入poe_results_list，格式为(N,3)：(L_in_physical_units, avrg_POE, valid_rays)
            poe_results_list.append((L_in_voxel_units * self.voxel_size, avrg_POE, valid_rays))
            # 输出当前进度
            if i % 10 == 0 or i == len(lengths_list) - 1:
                print(f"✔️[{i+1}/{len(lengths_list)}] 路径长度 {L_in_voxel_units} 体素单位；POE = {avrg_POE:.4f} (有效光线数: {valid_rays})")

        poe_results_list = np.array(poe_results_list)
        print("🎉消光概率计算完成！")
        self.poe_results_list = poe_results_list
        
        # 4. 曲线拟合，提取消光系数
        extinction_coefficient = self._extinction_curve_fit()
        self.extinction_coefficient = extinction_coefficient
        # 5.后处理相关函数
        
        return extinction_coefficient

# region 流程计算函数          
    # def _compute_ice_absorption(self, wavelength_nm, n_imag = 1.3e-5):
    #     '''
    #     根据波长计算冰的吸收系数。
        
    #     :param wavelength_nm: 波长 (nm)
    #     :param n_imag: 冰的折射率虚部
        
    #     :return ice_absorption: 冰的吸收系数
    #     '''
        
    #     wavelength_m = wavelength_nm * 1e-9  # 转换为米
    #     ice_absorption = 4 * np.pi * n_imag / wavelength_m  #
    #     return ice_absorption
      
    def _extinction_curve_fit(self, init_guess = 1500.0):
        '''
        对消光概率结果进行曲线拟合，提取消光系数。
        
        :param poe_results: 消光概率结果数组，形状为 (N, 3)，每行包含 (L_in_physical_units, avrg_POE, valid_rays)
        :param init_guess: 初始猜测的消光系数
        
        :return extinction_coefficient: 拟合得到的消光系数
        '''
        # 理想消光概率模型
        def ideal_extinction_model(L, extinction_coefficient):
            return 1.0 - np.exp(-extinction_coefficient * L)
        
        try:
            from scipy.optimize import curve_fit
        except Exception as e:
            print("❌ 未安装 scipy 库，无法进行曲线拟合。请安装 scipy 后重试。")
            return None
        
        print("🔍开始进行曲线拟合，提取消光系数...")
        
        try:
            popt, pcov = curve_fit(
                ideal_extinction_model,
                self.poe_results_list[:, 0],  # L_in_physical_units
                self.poe_results_list[:, 1],  # avrg_POE
                p0=[init_guess],
                bounds=(0, np.inf)
            )
            extinction_coefficient = popt[0]
            print("=" * 50)
            print(f"🎉 曲线拟合成功！拟合得到的消光系数为: {extinction_coefficient:.3f} m^-1")
            print("=" * 50)
            return extinction_coefficient
        except Exception as e:
            print(f"❌ 曲线拟合失败：{e}")
            return 0.0
   
# endregion
        
# region 后处理|可视化函数
    def plot_results(self):
        '''
        绘制消光概率计算结果及拟合曲线。
        '''
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("❌ 未安装 matplotlib 库，无法绘制结果。请安装 matplotlib 后重试。")
            return
        
        if self.poe_results_list is None:
            print("❌ 尚未进行消光概率计算，请先运行 run_simulation 方法。")
            return
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.poe_results_list[:, 0] * 1000, self.poe_results_list[:, 1], c='black', label='Monte Carlo Data', s=20)
        
        # 绘制拟合曲线
        L_smooth = np.linspace(0, max(self.poe_results_list[:, 0]), 100)
        pe_fit = 1 - np.exp(-self.extinction_coefficient * L_smooth)
        plt.plot(L_smooth * 1000, pe_fit, 'r--', label=f'Fit: $1 - e^{{-{self.extinction_coefficient:.1f} L}}$')
        
        plt.xlabel('Distance (mm)')
        plt.ylabel('Probability of Extinction (POE)')
        plt.title(f'Extinction Coefficient Simulation---$\\kappa_e = {self.extinction_coefficient:.1f}$')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # 保存图片到 Results 目录
        output_path = f"Results/extinction_coefficient_{self.extinction_coefficient:.0f}.png"
        os.makedirs("Results", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"📊 结果图已保存至: {output_path}")
        plt.show()
# endregion