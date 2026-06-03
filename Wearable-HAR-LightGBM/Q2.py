import lightgbm as lgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from prepare_the_data import X_train_scaled, y_train_adjusted, X_test_scaled, y_test_adjusted, activity_labels
# 构建LightGBM分类器
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=6,
    learning_rate=0.05,
    n_estimators=200,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
# 训练模型
lgb_model.fit(X_train_scaled, y_train_adjusted.values.ravel())
# 在测试集上进行预测 
y_pred = lgb_model.predict(X_test_scaled)
# 模型评估
accuracy = accuracy_score(y_test_adjusted, y_pred)
print(f"模型在测试集上的总体准确率: {accuracy:.4f}\n")
print("分类预测报告:")
print(classification_report(y_test_adjusted, y_pred, target_names=[activity_labels[i] for i in range(6)]))
# 绘制混淆矩阵热力图
cm = confusion_matrix(y_test_adjusted, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[activity_labels[i] for i in range(6)],
            yticklabels=[activity_labels[i] for i in range(6)])
plt.title('活动识别混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.xticks(rotation=45)
plt.show()