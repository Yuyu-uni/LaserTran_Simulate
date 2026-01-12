# LaserTran_Simulate

## 项目简介 (Project Overview)

**LaserTran_Simulate** 是一个基于 Python 的代码复现工作。该项目主要用于模拟雪的微观结构（Snow Microstructure），并研究激光或光子在复杂介质中的传输模拟。

## 理论基础 (Theoretical Basis)
核心算法参考了以下文献：
> **Xiong, C., et al.** "Modeling of Metric and Rayleigh Wave Propagation in Snow With a Bicontinuous Medium." *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, 2015.
> 
## TODOlist1: 双连续介质模拟生成
基于 **Gaussian Random Fields** 方法生成随机场，并通过阈值分割实现双相（冰/空气）介质的构建。

通过叠加大量随机方向和相位的余弦波来构建标量场 $S(\mathbf{r})$，然后根据目标体积分数 $f_v$ 进行二值化处理。

### 主要特性 (Features)

*   **物理参数控制**：支持自定义平均波数、粒径分布参数 ($b$) 和冰体积分数 ($f_v$)。
*   **内存优化**：实现了分块（Batch/Chunk）计算逻辑，支持在高分辨率（如 $128^3$ 或更高）下生成介质，避免内存溢出。
*   **缓存系统**：自动保存生成的随机场数据到 `RawData/` 目录，支持快速加载已生成的介质，避免重复计算。
*   **参数化文件命名**：缓存文件名包含所有关键参数（N、k、b、fv、L、分辨率、seed），方便区分不同配置。
*   **可视化**：自动生成并保存介质的切片图像，直观展示微观结构。
*   **Headless 支持**：针对 Linux 服务器/无头环境进行了优化，无需图形界面即可运行。


### 性能优化

**缓存加速**：首次生成随机场后，程序会自动保存到 `RawData/` 目录。后续使用相同参数运行时：
*   ✅ **首次生成**：需要几分钟（取决于分辨率和 N 值）
*   ✅ **缓存加载**：仅需 1-2 秒（提速 100+ 倍）

可通过 `FORCE_REGENERATE = True` 强制重新生成。


### 复现结果

运行完成后，程序在 `Results/` 目录下生成切片预览图：
*   `Results/snow_microstructure.png`

同时控制台会输出实际生成的体积分数统计。
**生成结果与论文符合较好。**

### 示例参数

默认配置用于模拟典型的雪微观结构：
*   尺寸 $L = 5\text{ mm}$
*   分辨率 $N = 128$ (体素大小 $\approx 40\mu m$)
*   平均波数 $\bar{k} = 5349.7 \text{ m}^{-1}$
*   体积分数 $f_v = 0.194$


## 文件结构 (File Structure)

*   `bicontinuous_medium.py`: 核心类 `BicontinuousMedium`，包含随机场生成、分块计算、二值化逻辑以及缓存保存/加载功能。
*   `main.py`: 主程序，设置物理参数，调用生成器并保存结果。支持智能缓存检测。
*   `RawData/`: 随机场数据缓存目录（自动生成，已添加到 `.gitignore`）。
*   `Results/`: 输出结果目录，存储切片可视化图像。
*   `pixi.toml`: 项目依赖配置文件。


## Git 分支说明 (Git Branching)

本项目当前包含以下主要分支：

*   **`main`**: 主分支，包含稳定的核心代码，运行于Windows平台，可以得到更好的性能和更大内存。
*   **`DevOnLinux`**: 位于 Linux 环境的开发分支。


## 环境依赖 (Dependencies)

本项目使用 [Pixi](https://prefix.dev/) 进行包管理。主要依赖包括：

*   `python >= 3.13`
*   `numpy` (数值计算)
*   `scipy` (特殊函数 `erfinv`, `gamma`)
*   `matplotlib` (绘图)
*   `mypy` & `scipy-stubs` (类型检查)
