import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from Q2 import lgb_model
from prepare_the_data import X_test_scaled, y_test_adjusted
def add_gaussian_noise(data, noise_level):
    """
    向数据中添加高斯噪声
    noise_level (sigma): 噪声强度，控制标准差
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    return data + noise
# 定义一系列噪声强度
noise_levels = np.linspace(0, 2.0, 10)
robustness_accuracies = []
# 测试不同噪声下的精度变化
for sigma in noise_levels:
    # 向标准化后的测试集添加噪声
    noisy_X_test = add_gaussian_noise(X_test_scaled, sigma)
    # 使用之前训练好的模型进行预测
    noisy_pred = lgb_model.predict(noisy_X_test)
    # 记录准确率
    acc = accuracy_score(y_test_adjusted, noisy_pred)
    robustness_accuracies.append(acc)
    print(f"噪声强度 (sigma={sigma:.2f}) -> 模型准确率: {acc:.4f}")
# 绘制鲁棒性曲线图
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, robustness_accuracies, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('模型预测鲁棒性分析：准确率随噪声强度的变化')
plt.xlabel('高斯噪声强度 (标准差 $\sigma$)')
plt.ylabel('模型预测准确率')
plt.grid(True, linestyle='--')
plt.axhline(y=robustness_accuracies[0]*0.8, color='r', linestyle='--', label='80% 初始性能基线')
plt.legend()
plt.show()
