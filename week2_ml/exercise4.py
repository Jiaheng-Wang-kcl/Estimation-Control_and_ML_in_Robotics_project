import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 生成数据
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# 扁平化数组
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 AdaBoost 回归模型并拟合
ada_regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=30),
    n_estimators=100,
    random_state=42,
    loss='linear'
)

ada_regressor.fit(X_train, y_train)

# 计算在训练集上的预测误差
y_train_pred = ada_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"AdaBoost Regressor Training MSE: {mse_train:.4f}, R2: {r2_train:.4f}")

# 绘制训练预测 vs. 实际值图
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'AdaBoost Regressor Training Predicted vs. Actual Values (MSE: {mse_train:.4f}, R2: {r2_train:.4f})')
plt.grid(True)
plt.show()
