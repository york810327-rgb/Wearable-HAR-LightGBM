import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import lightgbm as lgb
from prepare_the_data import X_train_scaled, X_test_scaled, y_train_adjusted
import pandas as pd
# PCA降维分析
pca = PCA().fit(X_train_scaled)
# 计算累计方差贡献率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# 绘制碎石图
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance, linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 累计方差')
plt.xlabel('主成分个数')
plt.ylabel('累计方差贡献率')
plt.title('PCA 降维：主成分个数与累计方差贡献率关系')
plt.legend()
plt.grid(True)
plt.show()
# 假设保留 95% 的信息量
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"保留 95% 方差所需的主成分个数为: {n_components_95}")
# 应用 PCA 降维
pca_optimal = PCA(n_components=n_components_95)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)
# 特征重要性分析
# 这里用树模型跑一遍原始数据，提取 Top 20 特征
lgb_explainer = lgb.LGBMClassifier(random_state=42, n_estimators=100)
lgb_explainer.fit(X_train_scaled, y_train_adjusted.values.ravel())
feature_importances = pd.Series(lgb_explainer.feature_importances_)
top_features = feature_importances.nlargest(20)
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title('Top 20 关键特征重要性排名 (基于 LightGBM)')
plt.xlabel('特征重要性得分')
plt.ylabel('原始特征索引')
plt.gca().invert_yaxis()  
plt.show()