from bicontinuous_medium import BicontinuousMedium
from extinction_calculator import ExtinctionCalculator
from absorption_calculator import AbsorptionCalculator
from phase_matrix_calculator import PhaseMatrixCalculator
import numpy as np
from scipy.special import erfinv, gamma 
import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端以支持无显示环境下的绘图

import matplotlib.pyplot as plt
import os

# ========== 配置参数 ==========
PARAMS = {
    'N': 1000,                    # 蒙特卡洛叠加次数
    'mean_waveNumber': 5349.7,    # 平均波数
    'b': 1.345,                   # 粒径分布参数
    'fv': 0.194,                  # 冰的体积分布
    'L': 0.01,                   # 介质物理尺寸 10mm
    'grid_resolution': 256,       # 介质网格分辨率
    'seed': 42,                    # 随机种子
    'RAW_DATA_DIR': "RawData",          # 随机场数据保存目录
    'FORCE_REGENERATE': False          # 设为 True 强制重新生成
}


# ============================

def main():
    # 定义双连续介质参数
    snow_medium = BicontinuousMedium(
        N=PARAMS['N'],
        mean_waveNumber=PARAMS['mean_waveNumber'],
        b=PARAMS['b'],
        fv=PARAMS['fv']
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
    # print(f"🚀目标体积分数: {snow_medium.fv}")
    # print(f"🌟实际体积分数: {actual_fv:.4f}")
    
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
    ext_result = ext_calc.run_simulation(wavelength_nm = 1300)
    # ext_calc.plot_results()
    
    abs_calc = AbsorptionCalculator(medium_instance=snow_medium, extinction_coefficient = ext_result)
    abs_result = abs_calc.run_simulation(wavelength_nm = 1300, max_dist_mm=20.0)
    abs_calc.plot_results()

    kappa_s = ext_result - abs_result
    pm_calc = PhaseMatrixCalculator(medium_instance=snow_medium, kappa_s=kappa_s)
    pm_calc.run_simulation(n_rays=100000, n_angle_bins=180)
    pm_calc.plot_results(wavelength_nm=1300)
    pm_calc.plot_cartesian(wavelength_nm=1300)
    pm_calc.get_asymmetry_parameter()
    
if __name__ == "__main__":
    main()
