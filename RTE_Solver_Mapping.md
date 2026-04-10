# 矢量辐射传输方程求解实现说明

本文档说明 [radiative_transfer_solver.py](radiative_transfer_solver.py) 的实现如何对应 [ .agent/MD_EqSolveGuide.md ] 中的推导步骤，并给出在 [main.py](main.py) 中的调用链。

## 1. 输入与前置量

当前代码的输入来自已有模块：

1. 消光系数 `kappa_e`
2. 相矩阵离散结果 `P11(Theta), P22(Theta), Theta`

在 [main.py](main.py) 中：

1. `ExtinctionCalculator.run_simulation` 计算 `ext_result`
2. `AbsorptionCalculator.run_simulation` 计算 `abs_result`
3. `kappa_s = kappa_e - kappa_a`
4. `PhaseMatrixCalculator.run_simulation` 输出 `P11, P22, angles`
5. 将 `ext_result, P11, P22, angles` 传入 `RadiativeTransferSolver`

## 2. 与推导公式的代码对应

### 2.1 漫射项方程与离散化

推导式（MD 文档）：

- 连续形式：
  `mu dI/dz = -kappa_e I + integral(P I) + S`
- 离散后：
  `mu_j dI(mu_j)/dz = -kappa_e I(mu_j) + sum_k w_k P(mu_j,mu_k) I(mu_k) + P(mu_j,mu0) I0 exp(-kappa_e z/mu0)`

代码对应：

1. `RadiativeTransferSolver._build_system_matrix`
   - 构建 `-kappa_e E + sum_k w_k P_jk` 的块矩阵
   - 再左乘 `M^{-1}` 得到 `B`
2. `RadiativeTransferSolver._build_source_vector`
   - 构建 `Q = P(mu_j,mu0) I0`
   - 再左乘 `M^{-1}` 得到 `F`

### 2.2 傅里叶展开（方位角解耦）

推导式（MD 文档）：

- `P0, Pmc, Pms` 由对 `phi` 的积分得到

代码对应：

1. `RadiativeTransferSolver._compute_pair_fourier`
   - 用 Gauss-Legendre 在 `[0, 2pi]` 上积分
   - 对每个 `(mu_out, mu_in)` 计算：
     - `P0 = (1/2pi) sum w_i P(phi_i)`
     - `Pmc = (1/pi) sum w_i P(phi_i) cos(m phi_i)`
     - `Pms = (1/pi) sum w_i P(phi_i) sin(m phi_i)`
2. `RadiativeTransferSolver._build_phase_fourier_tables`
   - 构建离散流 `mu_j, mu_k` 的完整系数表
3. `RadiativeTransferSolver._build_incident_phase_fourier`
   - 构建源项中的 `P(mu_j; mu0)` 傅里叶系数

说明：当前项目相矩阵输入仅有 `P11, P22`，代码采用

- `P_12 = diag(P11, P22, 0, 0)`

这是与现有数据接口一致的可运行版本，后续若你补充完整 4x4 相矩阵，可直接扩展对应函数。

### 2.3 标准 ODE 形式

推导式：

- `dI/dz = B I + F exp(-kappa_e z/mu0)`

代码对应：

- `RadiativeTransferSolver._solve_single_mode_surface`

其中包含：

1. 特解系数
   - `C = -(B + (kappa_e/mu0)E)^{-1} F`
2. 齐次解特征分解
   - `B v = lambda v`
3. 边界条件
   - 地表 `z=0`，对 `mu<0` 方向设置漫射入射为 0
   - 解线性方程得到待定系数
4. 最终得到 `z=0` 的模态解

### 2.4 模态重建

推导式：

- `I(mu,phi,0) = I0(mu) + sum_m [I_mc cos(m(phi-phi0)) + I_ms sin(m(phi-phi0))]`

代码对应：

1. `RadiativeTransferSolver.evaluate_surface_stokes`
2. `RadiativeTransferSolver._interp_mode_by_mu`

## 3. 物理量导出接口

### 3.1 BRDF

推导式：

- `BRDF = (I1 + I2) / mu0`

代码对应：

- `RadiativeTransferSolver.compute_brdf`
- `RadiativeTransferSolver.export_reflectance_products`（网格批量计算）

### 3.2 Albedo

推导式：

- `alpha(theta0) = integral_hemisphere BRDF * mu dOmega`

代码对应：

- `RadiativeTransferSolver.compute_albedo`

实现时利用方位角全积分后高阶模态平均为 0 的性质，仅使用 `m=0` 模态完成积分。

### 3.3 ARF

论文定义：

- `ARF = BRF / albedo = pi * BRDF / albedo`

代码对应：

- `RadiativeTransferSolver.compute_arf_from_brdf`
- `RadiativeTransferSolver.export_reflectance_products`

## 4. 可视化接口

### 4.1 BRDF / ARF / 偏振BRDF 极坐标图

代码对应：

- `RadiativeTransferSolver.plot_reflectance_products`

输出：

1. BRDF 极坐标分布
2. ARF 极坐标分布
3. 偏振 BRDF 极坐标分布
4. 主平面剖面曲线（phi=0 和 phi=180）

默认保存到：

- `Results/rte_reflectance.png`

### 4.2 Albedo-入射角曲线

代码对应：

- `RadiativeTransferSolver.plot_albedo_vs_sza`

默认保存到：

- `Results/rte_albedo_vs_sza.png`

## 5. main.py 调用顺序

在 [main.py](main.py) 中，新增调用流程如下：

1. 先算 `kappa_e, kappa_a, kappa_s`
2. 相矩阵模块输出 `P11, P22, angles`
3. `rte_solver = RadiativeTransferSolver(...)`
4. `rte_solver.run_simulation(...)`
5. `products = rte_solver.export_reflectance_products(...)`
6. `rte_solver.plot_reflectance_products(...)`
7. （可选）`rte_solver.plot_albedo_vs_sza(...)`

## 6. 当前实现边界与后续可扩展点

当前版本是可运行、可导出、可视化的完整链路，但有两个可继续增强点：

1. 相矩阵完整性
   - 目前仅使用 `P11, P22` 构建对角相矩阵
   - 若后续有完整 4x4 角散射矩阵，可扩展至全偏振耦合
2. 边界条件与多层结构
   - 当前使用半无限介质近似 + 顶边界无向下漫射入射
   - 可扩展为有限厚度雪层、底边界反射等更复杂条件
