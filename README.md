# LaserTran_Simulate

## 项目简介 (Project Overview)

**LaserTran_Simulate** 是一个基于 Python 的仿真工具，用于生成三维双连续随机介质（Bicontinuous Random Media）。该项目主要用于模拟雪的微观结构（Snow Microstructure），为激光或光子在复杂介质中的传输模拟提供几何模型。

本项目基于 **Gaussian Random Fields (GRF)** 方法生成随机场，并通过阈值分割实现双相（冰/空气）介质的构建。

## 理论基础 (Theoretical Basis)

核心算法参考了以下文献：
> **Xiong, C., et al.** "Modeling of Metric and Rayleigh Wave Propagation in Snow With a Bicontinuous Medium." *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, 2015.

通过叠加大量随机方向和相位的余弦波来构建标量场 $S(\mathbf{r})$，然后根据目标体积分数 $f_v$ 进行二值化处理。

## 主要特性 (Features)

*   **物理参数控制**：支持自定义平均波数 ($ar{k}$)、粒径分布参数 ($b$) 和冰体积分数 ($f_v$)。
*   **内存优化**：实现了分块（Batch/Chunk）计算逻辑，支持在高分辨率（如 $128^3$ 或更高）下生成介质，避免内存溢出（OOM）。
*   **可视化**：自动生成并保存介质的切片图像，直观展示微观结构。
*   **Headless 支持**：针对 Linux 服务器/无头环境进行了优化，无需图形界面即可运行。
*   **类型检查**：集成了 `mypy` 静态类型检查。

## 环境依赖 (Dependencies)

本项目使用 [Pixi](https://prefix.dev/) 进行包管理。主要依赖包括：

*   `python >= 3.14`
*   `numpy` (数值计算)
*   `scipy` (特殊函数 `erfinv`, `gamma`)
*   `matplotlib` (绘图)
*   `mypy` & `scipy-stubs` (类型检查)

## 快速开始 (Quick Start)

### 1. 安装环境

如果你已经安装了 `pixi`，在项目根目录下运行：

```bash
pixi install
```

### 2. 运行仿真

直接运行 `main.py` 即可生成微观结构：

```bash
pixi run python main.py
```

或者使用环境内的解释器：

```bash
.pixi/envs/default/bin/python main.py
```

### 3. 查看结果

运行完成后，程序会在 `Results/` 目录下生成切片预览图：
*   `Results/snow_microstructure.png`

同时控制台会输出实际生成的体积分数统计。

## 文件结构 (File Structure)

*   `bicontinuous_medium.py`: 核心类 `BicontinuousMedium`，包含随机场生成、分块计算和二值化逻辑。
*   `main.py`: 主程序，设置物理参数，调用生成器并保存结果。
*   `pixi.toml`: 项目依赖配置文件。

## 示例参数

默认配置用于模拟典型的雪微观结构：
*   尺寸 $L = 5\text{ mm}$
*   分辨率 $N = 128$ (体素大小 $\approx 40\mu m$)
*   平均波数 $\bar{k} = 5349.7 \text{ m}^{-1}$
*   体积分数 $f_v = 0.194$

```