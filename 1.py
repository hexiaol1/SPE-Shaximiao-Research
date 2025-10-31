import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('孔隙度+测井参数.txt', sep='\t')  # 假设数据文件名为data.txt

# 2. 准备数据
X = df.drop('孔隙度', axis=1)  # 特征矩阵
y = df['孔隙度']              # 目标变量

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 定义随机森林模型
rf = RandomForestRegressor(random_state=42, oob_score=True)

# 5. 设置超参数搜索范围
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 6. 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# 7. 输出最佳参数
print(f'最佳参数: {grid_search.best_params_}')

# 8. 使用最佳参数训练最终模型
best_rf = grid_search.best_estimator_

# 9. 预测测试集
y_pred = best_rf.predict(X_test)

# 10. 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')
print(f'OOB Score: {best_rf.oob_score_:.4f}')

# 11. 特征重要性分析
feature_importance = pd.Series(best_rf.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print("\n特征重要性：")
print(feature_importance)
# 12. 可视化部分

# 特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.xlabel("特征重要性")
plt.ylabel("特征")
plt.title("随机森林特征重要性")
plt.show()

# 预测值 vs 真实值散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("真实孔隙度")
plt.ylabel("预测孔隙度")
plt.title("预测值 vs 真实值")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 残差图
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("残差")
plt.title("残差分布")
plt.show()