# LaserTran_Simulate

## 项目简介 (Project Overview)

**LaserTran_Simulate** 是一个基于 Python 的科学计算项目，致力于复现雪微观结构（Snow Microstructure）的物理模型，并在此基础上模拟激光或光子在复杂随机介质中的传输过程。

本项目重点在于：
1.  **介质生成**：基于高斯随机场（Gaussian Random Fields）构建双连续介质模型。
2.  **光学特性模拟**：通过蒙特卡洛射线追踪（Monte Carlo Ray Tracing）方法，结合 DDA（Digital Differential Analyzer）算法，精确计算介质的**消光系数**和**吸收系数**。

## 理论基础 (Theoretical Basis)
核心算法及物理模型参考了以下文献：
> **Xiong, C., et al.** "Modeling of Metric and Rayleigh Wave Propagation in Snow With a Bicontinuous Medium." *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, 2015.

## 核心功能 (Core Features)

### 1. 双连续介质模拟 (Bicontinuous Medium Generation)
基于 **Gaussian Random Fields** 方法生成随机场，通过叠加大量随机方向和相位的余弦波构建标量场 $S(\mathbf{r})$，并通过阈值分割实现双相（冰/空气）介质的构建。

*   **物理参数控制**：支持自定义平均波数 ($k$)、粒径分布参数 ($b$) 和冰体积分数 ($f_v$)。
*   **内存优化**：实现分块（Batch/Chunk）计算逻辑，支持在高分辨率（如 $256^3$ 或 $512^3$）下生成介质。
*   **智能缓存**：自动保存生成的随机场数据到 `RawData/`，支持快速加载。文件名包含所有关键参数，方便区分。
*   **可视化**：支持 2D 切片导出及 3D 结构交互式查看（基于 PyVista）。

### 2. 光学特性计算 (Optical Property Simulation)
利用蒙特卡洛方法模拟光子在介质中的传输，计算关键光学参数。

*   **消光系数 (Extinction Coefficient)**：
    *   **算法**：追踪大量光线在介质中的自由程，计算消光概率（Probability of Extinction, POE）。
    *   **DDA 算法**：采用 **Digital Differential Analyzer (DDA)** 算法在体素网格中进行精确、高效的光线步进追踪，避免了传统步进法的穿墙误差。
    *   **曲线拟合**：基于模拟数据拟合 $P(L) = 1 - e^{-\kappa_e L}$ 模型，提取消光系数。

*   **吸收系数 (Absorption Coefficient)**：
    *   **物理过程**：模拟光线在冰/空气界面的**菲涅尔反射 (Fresnel Reflection)** 和 **折射 (Refraction)**。
    *   **路径追踪**：记录光线在冰相中的总光程 $L_{ice}$，计算吸收概率。
    *   **曲线拟合**：基于模拟数据拟合 $P_{abs}(L) = 1 - e^{-\kappa_a L}$ 模型，提取吸收系数。

### 3. 高性能计算 (High Performance)
*   **Numba 加速**：核心射线追踪算法使用 `numba` 进行 JIT 编译和并行加速（Parallel Computing），充分利用多核 CPU 性能。
*   **无锁并行**：设计了线程局部累加器，避免并行计算中的竞争条件。

## 性能优化 (Performance)

*   **缓存加载**：
    *   ✅ **首次生成**：几分钟（取决于分辨率和 N 值）
    *   ✅ **缓存加载**：仅需 1-2 秒（提速 100+ 倍）
*   **射线追踪**：
    *   得益于 DDA 算法和 Numba 并行，追踪 50,000 条光线仅需数秒至数十秒。

## 快速开始 (Quick Start)

### 环境依赖
本项目使用 [Pixi](https://prefix.dev/) 进行环境和依赖管理。

```bash
# 安装依赖
pixi install

# 运行主程序
pixi run python main.py
```

### 参数配置
在 `main.py` 中修改 `PARAMS` 字典即可调整模拟参数：

```python
PARAMS = {
    'N': 1000,                    # 蒙特卡洛叠加次数
    'mean_waveNumber': 5349.7,    # 平均波数 (m^-1)
    'b': 1.345,                   # 粒径分布参数
    'fv': 0.194,                  # 冰的体积分数
    'L': 0.01,                    # 介质物理尺寸 (m)
    'grid_resolution': 256,       # 介质网格分辨率
    'seed': 42,                   # 随机种子
    # ...
}
```

## 文件结构 (File Structure)

*   `main.py`: 主程序入口，负责参数配置、调用生成与计算模块、输出结果。
*   `bicontinuous_medium.py`: **[核心]** 介质生成类，包含随机场生成、二值化、缓存管理及可视化。
*   `extinction_calculator.py`: **[核心]** 消光系数计算模块，包含 DDA 射线追踪算法。
*   `absorption_calculator.py`: **[核心]** 吸收系数计算模块，包含菲涅尔反射/折射模拟。
*   `basic_utils.py`: 基础工具函数库（物理常数、向量运算、菲涅尔公式等）。
*   `RawData/`: 随机场数据缓存目录（自动生成）。
*   `Results/`: 结果输出目录（包含可视化图片、HTML 3D 模型）。
*   `pixi.toml`: 项目依赖配置文件。

## Git 分支说明 (Git Branching)

*   **`main`**: 主分支，包含稳定的核心代码，推荐在 **Windows (win64)** 平台运行以获得最佳性能。
*   **`DevOnLinux`**: 针对 Linux 环境（如 WSL）的开发分支，适配无头环境可视化。

## 结果示例

程序运行后将在 `Results/` 目录生成如下内容：
1.  **介质切片图** (`snow_microstructure.png`)：直观展示雪的微观结构。
2.  **消光系数拟合图** (`extinction_coefficient_*.png`)：展示 POE 数据点与拟合曲线。
3.  **吸收系数拟合图** (`absorption_coefficient_*.png`)：展示吸收概率数据与拟合曲线。