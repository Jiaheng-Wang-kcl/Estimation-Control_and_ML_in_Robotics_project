import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
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

# 创建 Bagging 回归模型并拟合
bagging_regressor = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=30),
    n_estimators=100,
    random_state=42
)

bagging_regressor.fit(X_train, y_train)

# 进行预测
y_pred = bagging_regressor.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Bagging Regressor MSE: {mse:.4f}, R2: {r2:.4f}")

# 绘制预测 vs. 实际值图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Bagging Regressor Predicted vs. Actual Values (MSE: {mse:.4f}, R2: {r2:.4f})')
plt.grid(True)
plt.show()
