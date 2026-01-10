import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 设定参数
N = 10000        # 采样数量多一点，直方图才平滑
mean_zeta = 5000 # 假设均值
b = 5            # 假设 b 参数

# 1. 代码中的采样 (Sampling) - 得到具体的 zeta 值
# 这对应我们 generate 方法里的一步
shape = b + 1
scale = mean_zeta / (b + 1)
zeta_samples = np.random.gamma(shape, scale, N)

print(f"前5个生成的 zeta 值: {zeta_samples[:5]}")
# 输出示例: [4823.1, 5102.5, 4200.3, ...] -> 这就是我们要用的数值！

# 2. 论文公式 (PDF) - 计算理论概率密度
x = np.linspace(0, 2 * mean_zeta, 1000)
# 使用 scipy.stats.gamma.pdf 来计算理论曲线
# 注意 scipy 的参数定义：a=shape, scale=scale
y_theoretical = gamma.pdf(x, a=shape, scale=scale)

# 3. 画图对比
plt.figure(figsize=(10, 6))

# 画出代码生成值的直方图 (归一化)
plt.hist(zeta_samples, bins=50, density=True, alpha=0.6, color='blue', label='Code Sampling (np.random.gamma)')

# 画出论文公式的理论曲线
plt.plot(x, y_theoretical, 'r-', linewidth=2, label='Paper Eq.(2) Theoretical PDF')

plt.title(f'Verification: Sampling vs Theory (mean={mean_zeta}, b={b})')
plt.xlabel('Wavenumber (zeta)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()