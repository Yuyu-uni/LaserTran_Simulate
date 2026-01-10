from bicontinuous_medium import BicontinuousMedium
import numpy as np
from scipy.special import erfinv, gamma 
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端以支持无显示环境下的绘图

import matplotlib.pyplot as plt

def main():
    # 定义双连续介质参数
    snow_medium = BicontinuousMedium(
        N=1000,
        mean_waveNumber=5349.7,  # 平均波数
        b=1.345,                 # 粒径分布参数
        fv=0.194                 # 冰的体积分布
    )
    
    medium = snow_medium.generate(
        L=0.005,                 # 介质物理尺寸 5mm (足以包含多个晶粒)
        grid_resolution=256,     # 介质网格分辨率 (确保每个晶粒有足够像素描述)
        seed=42,                 # 随机种子
        max_memory_gb=5.0       # 最大内存使用量 (GB)
    )
    
    # 比较体积分数的理论值和模拟值
    actual_fv = np.sum(medium) / medium.size
    print(f"🚀目标体积分数: {snow_medium.fv}")
    print(f"🌟实际体积分数: {actual_fv:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(snow_medium.get_slice_image(1), cmap='gray', interpolation='nearest')
    plt.title(f"Snow Microstructure Slice(fv={actual_fv:.3f})")
    plt.colorbar(label="Phase (0:Air, 1:Ice)")
    
    output_filename = "Results/snow_microstructure.png"
    plt.savefig(output_filename)
    print(f"Image saved to {output_filename}")
    # plt.show()
    
    
if __name__ == "__main__":
    main()
