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
        

# region 生成双连续介质核心方法
    def generate(self, L, grid_resolution, 
                  cache_dir=None, force_regenerate=False, seed=None, max_memory_gb=2.0):
        '''
        生成3D双连续介质（支持缓存加载）
        生成逻辑：
        1. 检查缓存目录中是否存在符合参数的文件
           - 如果存在且 fv 相同，直接加载并返回
           - 如果存在但 fv 不同，加载标量场并重新二值化
        2. 如果不存在，生成新的随机场并二值化
        
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
            match = self._find_existing_file(
                directory=cache_dir,
                N=self.N,
                mean_waveNumber=self.mean_waveNumber,
                b=self.b,
                fv=self.fv,
                L=L,
                resolution=grid_resolution,
                seed=seed
            )
            if match:
                if match["match"] == "exact":
                    print("=" * 50)
                    print("🚀 发现已存在文件（S(r)与fv均匹配），直接加载")
                    print("=" * 50)
                    self._load_from_file(match["path"])
                    return self.binary_medium
                
                elif match["match"] == "rebinarize":
                    print("=" * 50)
                    print("⚡ 发现相同的随机标量场 S(r)，但 fv 不同，重新二值化")
                    print("=" * 50)
                    # 只载入标量场与几何参数，不覆盖当前的 fv
                    self._load_scalar_field_only(match["path"])
                    # 根据当前 fv 进行二值化
                    self._self_binarize()
                    # 保存为新的文件（带当前 fv）
                    self._save_to_file(directory=cache_dir, seed=seed)
                    return self.binary_medium
        
        # 开始生成新的随机场
        self._self_scalar_field_generate(seed=seed, max_memory_gb=max_memory_gb)
        
        # 进行二值化
        self._self_binarize()
        
        # 自动保存缓存
        if cache_dir:
            self._save_to_file(directory=cache_dir, seed=seed)
        
        # 返回二值化后的介质
        return self.binary_medium
    
    
    def _self_scalar_field_generate(self, seed=None, max_memory_gb=2.0):
        '''
        内部调用：生成全新随机标量场 S(r)
        
        :param seed: 随机种子
        :param max_memory_gb: 最大内存使用量（GB），用于控制分块大小
        '''
        
        print("=" * 50)
        print(f"⏳ 正在生成新的随机场S(r)... 关键参数: N={self.N}, mean_waveNumber={self.mean_waveNumber}, b={self.b}")
        print(f"⏳ 生成尺度：L={self.L}, resolution={self.resolution}")
        print("=" * 50)
        
        if seed is not None:
            np.random.seed(seed)
        
        # 生成坐标网格
        x = np.linspace(0, self.L, self.resolution)
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
        N_voxels = self.resolution ** 3
        
        # 计算分块大小
        # phases 矩阵大小为 (N, chunk_size)，每个元素 8 bytes (float64)
        # 我们还需要存储 cos(phases)，所以实际内存约为 2 * N * chunk_size * 8 bytes
        bytes_per_element = 8  # float64
        max_memory_bytes = max_memory_gb * 1024**3
        # 考虑 phases 和 cos(phases) 两个矩阵
        chunk_size = int(max_memory_bytes / (2 * self.N * bytes_per_element))
        chunk_size = max(1, min(chunk_size, N_voxels))  # 确保在有效范围内
        
        n_chunks = (N_voxels + chunk_size - 1) // chunk_size
        
        print("*" * 50)
        print(f"开始生成随机场 (分辨率:{self.resolution}^3={N_voxels}体素, 蒙特卡洛叠加次数:{self.N})...")
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
        self.scalar_field = S_flatten.reshape((self.resolution, self.resolution, self.resolution))
        del S_flatten
        
        print("✔️ 随机场生成完成。")
    
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
        
        print("=" * 50)
        print(f"⏳ 开始阈值分割:fv={self.fv}, 计算阈值={threshold:.4f}")        
        print("=" * 50)
        
        self.binary_medium = S_normalized > threshold  # True: 冰, False: 空气
        print("✔️ 二值化完成。")
# endregion 
        
# region 后处理 | 可视化相关方法            
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
    
    def visualize_3d(self, show_scalar_field=False, opacity=0.8, cmap="bone", 
                     show_edges=False, window_size=(1024, 1024), export_html=False, html_filename=None, auto_downsample=True, downsample_threshold=200):
        '''
        使用PyVista进行3D可视化，展示双连续介质结构
        
        :param show_scalar_field(bool): 是否显示连续标量场（而非二值化结果）
        :param opacity(float): 不透明度 (0-1)
        :param cmap(str): 颜色映射方案
        :param show_edges(bool): 是否显示网格边缘
        :param window_size(tuple): 窗口大小 (宽, 高)
        :param export_html(bool): 是否导出为HTML文件
        :param html_filename(str): 导出的HTML文件名
        :param auto_downsample(bool): 是否对高分辨率数据自动降采样以提升性能（默认True）
        '''
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("需要安装pyvista库: pixi add pyvista")
        
        # 导入数据
        if show_scalar_field:
            if self.scalar_field is None:
                raise ValueError("请先生成介质 (调用 generate 方法) 后再进行可视化。")
            data = self.scalar_field.astype(np.float32)
            title = f"标量场 S(r) - L={self.L*1000:.2f}mm, res={self.resolution}"
        else:
            if self.binary_medium is None:
                raise ValueError("请先生成介质 (调用 generate 方法) 后再进行可视化。")
            data = self.binary_medium.astype(np.float32)
            title = f"二值化介质 (fv={self.fv:.3f}) - L={self.L*1000:.2f}mm, res={self.resolution}"
        
        # 创建PyVista的UniformGrid (ImageData)
        grid = pv.ImageData()
        
        # 自动降采样逻辑：适用于HTML导出和交互式显示
        if auto_downsample and self.resolution > downsample_threshold:
            print(f"⚠️ 检测到高分辨率 ({self.resolution}^3)，为防止内存溢出和卡顿，将自动降采样至 {downsample_threshold}^3。")
            print("🌟(如需查看原始分辨率，请调用 visualize_3d(..., auto_downsample=False))")
            
            # 计算降采样因子
            factor = self.resolution // downsample_threshold
            
            # 使用简单的切片进行降采样
            data_downsampled = data[::factor, ::factor, ::factor]
            
            # 更新分辨率和网格尺寸
            dims = np.array(data_downsampled.shape)
            grid.dimensions = dims + 1
            
            # 更新spacing
            spacing = self.L / dims[0] # 假设各向同性
            grid.spacing = (spacing, spacing, spacing)
            
            grid.origin = (0, 0, 0)
            grid.cell_data["values"] = data_downsampled.flatten(order="F")
        else:
            grid.dimensions = np.array(data.shape) + 1  # 节点数 = 单元数 + 1
            
            # 设置物理尺寸
            spacing = self.L / self.resolution
            grid.spacing = (spacing, spacing, spacing)
            grid.origin = (0, 0, 0)
            
            # 添加数据到单元
            grid.cell_data["values"] = data.flatten(order="F")
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=window_size)
        
        if show_scalar_field:
            # 体积渲染通常在黑色背景下效果更好，且能避免"bone"等亮色map在白背景下不可见的问题
            plotter.set_background("black")
            
            # 将单元数据转换为点数据，体积渲染效果更平滑
            grid_pt = grid.cell_data_to_point_data()
            
            if export_html:
                # ⚠️ HTML导出通常不支持复杂的体积渲染(Volume Rendering)，会导致黑屏
                print("⚠️ HTML导出模式下，体积渲染可能不被支持。已自动切换为'多层等值面'显示以确保可见性。")                
                # 生成5个等值面，覆盖数据范围
                # 数据已标准化，大概在[-3, 3]之间
                rng = grid_pt.get_data_range()
                isosurfaces = grid_pt.contour(isosurfaces=5, rng=rng)
                
                plotter.add_mesh(isosurfaces, scalars="values", cmap=cmap, opacity=opacity, smooth_shading=True)
            else:
                # 连续标量场：使用体积渲染
                plotter.add_volume(grid_pt, scalars="values", cmap=cmap, opacity="sigmoid")
        else:
            plotter.set_background("white")
            # 二值化介质：只显示冰相 (值为1的部分)
            thresholded = grid.threshold(value=0.5, scalars="values")
            plotter.add_mesh(thresholded, color="lightblue", opacity=opacity, 
                           show_edges=show_edges, edge_color="gray")
        
        # 添加坐标轴和标题
        plotter.add_axes()
        plotter.add_title(title, font_size=12)
        
        # 添加边界框
        plotter.add_bounding_box(color="black", line_width=1)
        
        if export_html:
            if html_filename is None:
                # 默认文件名
                filename = self._get_filename() + ".html"
                html_filename = os.path.join("Results", filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(html_filename)), exist_ok=True)
            
            plotter.export_html(html_filename)
            print(f"✔️ 3D场景已导出为HTML文件: {html_filename}")
        else:
            # 显示
            plotter.show()
# endregion      

# region 文件保存与加载相关方法  
    def _get_filename(self, seed=None):
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
    
    def _save_to_file(self, directory="RawData", seed=None):
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
        
        filename = self._get_filename(seed)
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
        
        print(f"✔️ 随机场数据已保存到: {filepath}")
        return filepath
    
    def _load_from_file(self, filepath):
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
        
        print(f"✔️ 加载完成! 关键参数：N={self.N}, k={self.mean_waveNumber}, b={self.b}, fv={self.fv}")
        print(f"✔️ 生成尺度：L={self.L}, resolution={self.resolution}")
        return self
    
    def _load_scalar_field_only(self, filepath):
        '''
        仅从文件中加载随机标量场及其几何/生成参数，不修改当前 fv，也不加载二值化结果
        
        :param filepath(str): 文件路径
        :return: self (链式调用)
        '''
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        print(f"📂 正在加载随机标量场数据: {filepath}")
        data = np.load(filepath)
        
        # 仅恢复标量场和生成参数（不覆盖当前 fv）
        self.scalar_field = data['scalar_field']
        self.N = int(data['N'])
        self.mean_waveNumber = float(data['mean_waveNumber'])
        self.b = float(data['b'])
        self.L = float(data['L'])
        self.resolution = int(data['resolution'])
        
        print(f"✔️ 标量场加载完成! 关键参数：N={self.N}, k={self.mean_waveNumber}, b={self.b}")
        print(f"✔️ 生成尺度：L={self.L}, resolution={self.resolution}")
        
        print(f"将使用当前 fv={self.fv} 进行重新二值化")
        return self
    
    @staticmethod
    def _find_existing_file(directory="RawData", N=None, mean_waveNumber=None, b=None, fv=None, L=None, resolution=None, seed=None):
        '''
        查找符合参数的已存在文件
        分为两个阶段：
        1. 如果出现相同随机标量场S(r)，且fv相同，则认为是相同文件，直接导出
        2. 如果S(r)相同但fv不同，则仅需要重新二值化，保存为新文件
        
        :return: dict | None
            - {"path": <file>, "match": "exact"}: S(r)和fv都匹配，可直接加载
            - {"path": <file>, "match": "rebinarize"}: S(r)匹配但fv不同，需重新二值化
            - None: 未找到匹配文件
        '''
        if not os.path.exists(directory):
            return None
        
        # 阶段1: 构建精确匹配的文件名（包含 fv）
        exact_prefix = f"N{N}_k{mean_waveNumber:.1f}_b{b:.3f}_fv{fv:.3f}"
        if L is not None:
            exact_prefix += f"_L{L*1000:.2f}mm"
        if resolution is not None:
            exact_prefix += f"_res{resolution}"
        if seed is not None:
            exact_prefix += f"_seed{seed}"
        
        exact_file = os.path.join(directory, exact_prefix + ".npz")
        
        if os.path.exists(exact_file):
            return {"path": exact_file, "match": "exact"}
        
        # 阶段2: 构建随机场匹配的文件名前缀（不包含 fv）
        # 决定随机场 S(r) 的参数: N, mean_waveNumber, b, L, resolution, seed
        base_prefix = f"N{N}_k{mean_waveNumber:.1f}_b{b:.3f}_fv"
        base_suffix = ""
        if L is not None:
            base_suffix += f"_L{L*1000:.2f}mm"
        if resolution is not None:
            base_suffix += f"_res{resolution}"
        if seed is not None:
            base_suffix += f"_seed{seed}"
        
        # 搜索具有相同 S(r) 但不同 fv 的文件
        for fname in os.listdir(directory):
            if not fname.endswith(".npz"):
                continue
            # 检查前缀和后缀是否匹配（中间的 fv 值可以不同）
            if fname.startswith(base_prefix) and (base_suffix + ".npz" in fname):
                return {"path": os.path.join(directory, fname), "match": "rebinarize"}
        
        return None
    
# endregion