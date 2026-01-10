import numpy as np
from scipy.special import erfinv
import os
from pathlib import Path

class BicontinuousMedium:
    """
    A class to represent and simulate a bicontinuous medium.
    参考文献:Xiong et al., IEEE TGRS, 2015.
    """
    def __init__(self, N, mean_waveNumber, b, fv):
        '''
        Initialize the bicontinuous medium with given parameters.
        N (int): 蒙特卡洛叠加次数。建议1000
        mean_waveNumber (float): 平均波数
        b (float): 粒径分布参数
        fv (float): 冰的体积分数
        '''
        self.N = N
        self.mean_waveNumber = mean_waveNumber
        self.b = b
        self.fv = fv

        # 储存生成场的数据，防止重复计算
        self.scalar_field = None  # S(r)
        self.binary_medium = None # 二值化后的介质 (True: 冰, False: 空气)
        self.L = None             # 介质物理尺寸（立方体边长m）
        self.resolution = None    # 介质网格分辨率
        
    def calculate_theoretical_SSA(self) -> float:
        '''
        计算双连续介质的理论比表面积（SSA）
        
        Returns:
            SSA(float):理论比表面积值(m^-1)
        '''
        # 这里根据Xiong et al., IEEE TGRS, 2015的公式实现理论SSA计算
        # 具体公式需要根据文献确定，这里给出一个示例占位符
        # SSA = some_function_of(self.mean_zeta, self.b, self.fv)
        # return SSA
        return 0.1  # 占位符返回值

    def generate(self, L, grid_resolution, seed, max_memory_gb=2.0, 
                  cache_dir=None, force_regenerate=False):
        '''
        生成3D双连续介质（支持缓存加载）
        
        :param L(float): 立方体边长(m)
        :param grid_resolution(int): 网格分辨率（每边网格数）
        :param seed(int): 随机种子
        :param max_memory_gb(float): 最大内存使用量（GB），用于控制分块大小
        :param cache_dir(str): 缓存目录，如果指定则会检查/保存缓存文件
        :param force_regenerate(bool): 是否强制重新生成（忽略缓存）
        '''
        # 先设置参数以便生成文件名
        self.L = L
        self.resolution = grid_resolution
        
        # 检查是否存在缓存文件
        if cache_dir and not force_regenerate:
            existing_file = self.find_existing_file(
                directory=cache_dir,
                N=self.N,
                mean_waveNumber=self.mean_waveNumber,
                b=self.b,
                fv=self.fv,
                L=L,
                resolution=grid_resolution,
                seed=seed
            )
            if existing_file:
                print("=" * 50)
                print("🚀 发现已存在的随机场数据，使用快速加载模式")
                print("=" * 50)
                self.load_from_file(existing_file)
                return self.binary_medium
        
        # 开始生成新的随机场
        print("=" * 50)
        print("⏳ 正在生成新的随机场...")
        print("=" * 50)
        
        if seed is not None:
            np.random.seed(seed)
        
        # 生成坐标网格
        x = np.linspace(0, L, grid_resolution)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        # 展平以便向量化运算 (3, N_voxels)
        # 结果：每一列代表一个空间点的坐标
        '''
            r_vectors = [[x₀, x₁, x₂, ..., xₙ],   # 所有点的 x 坐标
                         [y₀, y₁, y₂, ..., yₙ],   # 所有点的 y 坐标
                         [z₀, z₁, z₂, ..., zₙ]]   # 所有点的 z 坐标
        '''
        r_vectors = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)  
        
        # 释放不再需要的网格变量
        del X, Y, Z

        # 计算波矢量（wave vector） 执行gamma抽样过程
        wave_numbers = np.random.gamma(self.b + 1, self.mean_waveNumber / (self.b + 1), self.N)
        # 生成沿球面随机方向的单位向量
        vectors = np.random.randn(3, self.N)
        # 归一化为单位向量
        vectors /= np.linalg.norm(vectors, axis=0)
        
        wave_vectors = wave_numbers * vectors  # (3, N)
        
        # 计算随机相位
        random_psi = np.random.uniform(0, 2 * np.pi, self.N)
        
        # 计算标量场 S(r) - 使用分块处理以节省内存--------------------------------------
        # S(r) = Σ cos(k_i · r + ψ_i)/√N
        N_voxels = grid_resolution ** 3
        
        # 计算分块大小
        # phases 矩阵大小为 (N, chunk_size)，每个元素 8 bytes (float64)
        # 我们还需要存储 cos(phases)，所以实际内存约为 2 * N * chunk_size * 8 bytes
        bytes_per_element = 8  # float64
        max_memory_bytes = max_memory_gb * 1024**3
        # 考虑 phases 和 cos(phases) 两个矩阵
        chunk_size = int(max_memory_bytes / (2 * self.N * bytes_per_element))
        chunk_size = max(1, min(chunk_size, N_voxels))  # 确保在有效范围内
        
        n_chunks = (N_voxels + chunk_size - 1) // chunk_size
        
        print(f"正在生成随机场 (分辨率:{grid_resolution}^3={N_voxels}体素, 蒙特卡洛叠加次数:{self.N})...")
        print(f"内存优化: 分{n_chunks}块处理，每块约{chunk_size}个体素")
        
        # 预分配结果数组
        S_flatten = np.zeros(N_voxels, dtype=np.float64)
        
        # 分块计算
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N_voxels)
            
            # 提取当前块的坐标
            r_chunk = r_vectors[:, start_idx:end_idx]  # (3, chunk_size)
            
            # 计算当前块的相位
            # phases_chunk shape: (N, chunk_size)
            phases_chunk = np.dot(wave_vectors.T, r_chunk) + random_psi[:, np.newaxis]
            
            # 计算并累加余弦值
            S_flatten[start_idx:end_idx] = np.sum(np.cos(phases_chunk), axis=0) / np.sqrt(self.N)
            
            # 显示进度
            if n_chunks > 1 and (i + 1) % max(1, n_chunks // 10) == 0:
                print(f"  进度: {100 * (i + 1) / n_chunks:.1f}%")
        
        # 释放不再需要的变量
        del r_vectors, wave_vectors, random_psi
        
        # 重塑为3D场 将平坦数组重塑为立方体
        self.scalar_field = S_flatten.reshape((grid_resolution, grid_resolution, grid_resolution))
        del S_flatten
        
        print("随机场生成完成。")
        
        # 进行二值化
        self._self_binarize()
        
        # 自动保存缓存
        if cache_dir:
            self.save_to_file(directory=cache_dir, seed=seed)
        
        # 返回二值化后的介质
        return self.binary_medium

    def _self_binarize(self):
        '''
        内部调用：根据体积分数 fv 对标量场进行二值化
        '''
        # 保证数值稳定性，将高斯随机场标准化为N(0,1)
        S = self.scalar_field
        S_mean = np.mean(S)
        S_std = np.std(S)
        S_normalized = (S - S_mean) / S_std
        
        # 计算阈值 threshold 使得 P(S > threshold) = fv
        # 注意：此时是标准正态分布下
        # 推导过程：
        # 1. 标准正态分布的CDF: Φ(x) = (1/2)[1 + erf(x/√2)]
        # 2. 要求 P(S > t) = fv，即 P(S ≤ t) = 1 - fv
        # 3. 因此 Φ(t) = 1 - fv
        # 4. (1/2)[1 + erf(t/√2)] = 1 - fv
        # 5. erf(t/√2) = 2(1 - fv) - 1 = 1 - 2*fv
        # 6. t/√2 = erfinv(1 - 2*fv)
        # 7. t = √2 * erfinv(1 - 2*fv)
        # 所以 √2 来源于标准正态分布CDF与误差函数erf的关系
        threshold = np.sqrt(2) * erfinv(1 - 2 * self.fv)
        
        
        print(f"开始阈值分割:fv={self.fv}, 计算阈值={threshold:.4f}")
        self.binary_medium = S_normalized > threshold  # True: 冰, False: 空气
        print("二值化完成。")
        
        
    def get_slice_image(self, axis = 2, index=None):
        '''
        获取二值化介质在指定轴向的切片图像
        
        :param axis(int): 轴向 (0:x, 1:y, 2:z)
        :param index(int): 切片索引 (默认中间切片)
        :return: 2D numpy array 切片图像
        '''
        if self.binary_medium is None:
            raise ValueError("请先生成介质 (调用 generate 方法) 后再获取切片图像。")
        
        if index is None:
            index = self.resolution // 2  # 默认中间切片
        
        if axis == 0:
            slice_image = self.binary_medium[index, :, :]
        elif axis == 1:
            slice_image = self.binary_medium[:, index, :]
        elif axis == 2:
            slice_image = self.binary_medium[:, :, index]
        else:
            raise ValueError("轴向参数 axis 必须为 0 (x), 1 (y), 或 2 (z)。")
        
        return slice_image
    
    
    def get_filename(self, seed=None):
        '''
        根据介质参数生成唯一的文件名
        
        :param seed(int): 随机种子
        :return: 文件名字符串 (不包含扩展名)
        '''
        # 格式: N{N}_k{mean_waveNumber}_b{b}_fv{fv}_L{L}_res{resolution}_seed{seed}
        filename = f"N{self.N}_k{self.mean_waveNumber:.1f}_b{self.b:.3f}_fv{self.fv:.3f}"
        if self.L is not None:
            filename += f"_L{self.L*1000:.2f}mm"
        if self.resolution is not None:
            filename += f"_res{self.resolution}"
        if seed is not None:
            filename += f"_seed{seed}"
        return filename
    
    def save_to_file(self, directory="RawData", seed=None):
        '''
        将生成的随机场和二值化介质保存到文件
        
        :param directory(str): 保存目录
        :param seed(int): 随机种子 (用于文件名)
        :return: 保存的文件路径
        '''
        if self.binary_medium is None:
            raise ValueError("请先生成介质 (调用 generate 方法) 后再保存。")
        
        # 确保目录存在
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        filename = self.get_filename(seed)
        filepath = os.path.join(directory, filename + ".npz")
        
        # 保存所有必要数据
        np.savez_compressed(
            filepath,
            scalar_field=self.scalar_field,
            binary_medium=self.binary_medium,
            N=self.N,
            mean_waveNumber=self.mean_waveNumber,
            b=self.b,
            fv=self.fv,
            L=self.L,
            resolution=self.resolution
        )
        
        print(f"✅ 随机场数据已保存到: {filepath}")
        return filepath
    
    def load_from_file(self, filepath):
        '''
        从文件加载已保存的随机场和二值化介质
        
        :param filepath(str): 文件路径
        :return: self (链式调用)
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        print(f"📂 正在加载随机场数据: {filepath}")
        data = np.load(filepath)
        
        # 恢复所有数据
        self.scalar_field = data['scalar_field']
        self.binary_medium = data['binary_medium']
        self.N = int(data['N'])
        self.mean_waveNumber = float(data['mean_waveNumber'])
        self.b = float(data['b'])
        self.fv = float(data['fv'])
        self.L = float(data['L'])
        self.resolution = int(data['resolution'])
        
        print(f"✅ 加载完成! 分辨率: {self.resolution}^3, 体积分数: {self.fv}")
        return self
    
    @staticmethod
    def find_existing_file(directory="RawData", N=None, mean_waveNumber=None, b=None, fv=None, L=None, resolution=None, seed=None):
        '''
        查找符合参数的已存在文件
        
        :return: 文件路径 (如果存在)，否则返回 None
        '''
        if not os.path.exists(directory):
            return None
        
        # 构建期望的文件名模式
        expected_prefix = f"N{N}_k{mean_waveNumber:.1f}_b{b:.3f}_fv{fv:.3f}"
        if L is not None:
            expected_prefix += f"_L{L*1000:.2f}mm"
        if resolution is not None:
            expected_prefix += f"_res{resolution}"
        if seed is not None:
            expected_prefix += f"_seed{seed}"
        
        expected_file = os.path.join(directory, expected_prefix + ".npz")
        
        if os.path.exists(expected_file):
            return expected_file
        return None