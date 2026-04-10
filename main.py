from bicontinuous_medium import BicontinuousMedium
from extinction_calculator import ExtinctionCalculator
from absorption_calculator import AbsorptionCalculator
from phase_matrix_calculator import PhaseMatrixCalculator
from radiative_transfer_solver import RadiativeTransferSolver
from basic_utils import convert_ssa_re_to_medium_params
import numpy as np
from scipy.special import erfinv, gamma
import matplotlib

# 优先尝试交互式后端，便于本地弹出图窗。
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt
import os

# ========== 配置参数 ==========
PARAMS = {
    'N': 1000,                    # 蒙特卡洛叠加次数
    'mean_waveNumber': 5349.7,    # 平均波数
    'b': 1.345,                   # 粒径分布参数
    'fv': 0.194,                  # 冰的体积分数 (snow density/ice density)
    'SSA': None,                  # 比表面积 (m^-1)，提供后可反推 mean_waveNumber
    'R_e': None,                  # 等效晶粒半径 (m)，可单独输入或与 SSA 联合输入
    'L': 0.01,                    # 介质物理尺寸 10mm
    'grid_resolution': 256,       # 介质网格分辨率
    'seed': 42,                   # 随机种子
    'RAW_DATA_DIR': "RawData",    # 随机场数据保存目录
    'FORCE_REGENERATE': False,    # 设为 True 时强制重新生成

    # 辐射传输求解参数
    'solar_zenith_deg': 53.0,     # 入射天顶角（论文中常用约 50-70 度）
    'solar_azimuth_deg': 0.0,    # 入射方位角
    'rte_n_streams': 64,          # 离散流数（必须为偶数）
    'rte_fourier_order': 32,      # 傅里叶展开最高阶
    'rte_phi_quadrature': 96,     # 方位角积分点数
    'RUN_ALBEDO_SZA_SCAN': False   # 是否额外扫描入射角-反照率曲线
}
# 来自 figure1 的雪层参数 (mean_waveNumber=5349.7, b=1.345, fv=0.194)

# ============================


def main():
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    medium_params = {
        'mean_waveNumber': PARAMS['mean_waveNumber'],
        'b': PARAMS['b'],
        'fv': PARAMS['fv'],
    }

    # 若提供了 SSA/R_e，则反推并覆盖 mean_waveNumber。
    if PARAMS.get('SSA') is not None or PARAMS.get('R_e') is not None:
        converted = convert_ssa_re_to_medium_params(
            ssa=PARAMS.get('SSA'),
            r_e=PARAMS.get('R_e'),
            b=PARAMS['b'],
            fv=PARAMS['fv'],
        )
        medium_params.update({
            'mean_waveNumber': converted['mean_waveNumber'],
            'b': converted['b'],
            'fv': converted['fv'],
        })
        print(
            "Using SSA/R_e converted params: "
            f"mean_waveNumber={converted['mean_waveNumber']:.4f}, "
            f"b={converted['b']:.4f}, fv={converted['fv']:.4f}, "
            f"SSA={converted['SSA']:.6e}, R_e={converted['R_e']:.6e}"
        )

    # 定义双连续介质参数
    snow_medium = BicontinuousMedium(
        N=PARAMS['N'],
        mean_waveNumber=medium_params['mean_waveNumber'],
        b=medium_params['b'],
        fv=medium_params['fv']
    )

    # 生成或加载随机场（自动检测缓存）
    snow_medium.generate(
        L=PARAMS['L'],
        grid_resolution=PARAMS['grid_resolution'],
        seed=PARAMS['seed'],
        cache_dir=PARAMS['RAW_DATA_DIR'],
        force_regenerate=PARAMS['FORCE_REGENERATE'],
        max_memory_gb=5.0
    )

    # 比较体积分数的理论值和模拟值
    # actual_fv = np.sum(snow_medium.get_binary_medium()) / snow_medium.get_binary_medium().size
    # print(f"目标体积分数: {snow_medium.fv}")
    # print(f"实际体积分数: {actual_fv:.4f}")

    # 可视化二维切片
    # plt.figure(figsize=(8, 8))
    # plt.imshow(snow_medium.get_slice_image(1), cmap='gray', interpolation='nearest')
    # plt.title(f"Snow Microstructure Slice(fv={actual_fv:.3f})")
    # plt.colorbar(label="Phase (0:Air, 1:Ice)")
    # output_filename = "Results/snow_microstructure.png"
    # plt.savefig(output_filename)
    # print(f"Image saved to {output_filename}")
    # plt.show()

    # 可视化三维结构
    # snow_medium.visualize_3d(show_scalar_field=False, display_mode="interact", auto_downsample=False)

    ext_calc = ExtinctionCalculator(medium_instance=snow_medium)
    ext_result = ext_calc.run_simulation(wavelength_nm=1300)
    # ext_calc.plot_results()

    abs_calc = AbsorptionCalculator(medium_instance=snow_medium, extinction_coefficient=ext_result)
    abs_result = abs_calc.run_simulation(wavelength_nm=1300, max_dist_mm=20.0)
    # abs_calc.plot_results()

    kappa_s = ext_result - abs_result
    pm_calc = PhaseMatrixCalculator(medium_instance=snow_medium, kappa_s=kappa_s)
    P11, P22, angles = pm_calc.run_simulation(n_rays=100000, n_angle_bins=180)
    # pm_calc.plot_results(wavelength_nm=1300)
    # pm_calc.plot_cartesian(wavelength_nm=1300)
    pm_calc.get_asymmetry_parameter()

    # =========================
    # 辐射传输方程求解 + BRDF/Albedo/ARF 导出
    # =========================
    rte_solver = RadiativeTransferSolver(
        extinction_coefficient=ext_result,
        phase_angles=angles,
        p11=P11,
        p22=P22,
        n_streams=PARAMS['rte_n_streams'],
        fourier_order=PARAMS['rte_fourier_order'],
        n_phi_quadrature=PARAMS['rte_phi_quadrature'],
    )

    rte_solver.run_simulation(
        solar_zenith_deg=PARAMS['solar_zenith_deg'],
        solar_azimuth_deg=PARAMS['solar_azimuth_deg'],
    )

    os.makedirs("Results", exist_ok=True)
    products = rte_solver.export_reflectance_products(
        output_npz_path="Results/rte_reflectance_products.npz"
    )

    rte_solver.plot_reflectance_products(
        products=products,
        output_prefix="Results/rte_reflectance",
        show=True,
    )

    # 可选：输出反照率-入射角关系
    if PARAMS['RUN_ALBEDO_SZA_SCAN']:
        sza_grid = np.linspace(10.0, 80.0, 100)
        rte_solver.plot_albedo_vs_sza(
            solar_zenith_deg_samples=sza_grid,
            output_path="Results/rte_albedo_vs_sza.png",
            show=True,
        )

    summary = rte_solver.summarize(products)
    print("=" * 58)
    print("RTE 求解完成，关键物理量：")
    print(f"   Albedo = {summary['albedo']:.6f}")
    print(f"   BRDF range = [{summary['brdf_min']:.6e}, {summary['brdf_max']:.6e}]")
    print(f"   ARF range  = [{summary['arf_min']:.6e}, {summary['arf_max']:.6e}]")
    print("=" * 58)


if __name__ == "__main__":
    main()
