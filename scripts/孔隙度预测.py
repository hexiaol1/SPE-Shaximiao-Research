import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load  # 用于保存和加载模型

# 设置中文显示和字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
data = pd.read_csv('孔隙度+测井参数.txt', sep='\t')

# 准备训练数据
X = data.drop(['孔隙度', '_GR', '_RT', '_CAL', '_RXO'], axis=1)
y = data['孔隙度']

# 训练模型
model = RandomForestRegressor(random_state=42)  # 实例化随机森林模型
model.fit(X, y)

# 保存模型到文件
dump(model, 'random_forest_model.joblib')  # 保存模型
print("模型已保存为 'random_forest_model.joblib'")

# 加载模型（可选，这里只是为了演示加载过程）
loaded_model = load('random_forest_model.joblib')

# 读取新数据文件
new_data = pd.read_csv('待预测孔隙度.txt', sep='\t')  # 假设新数据文件是制表符分隔的
print("新数据文件内容：")
print(new_data)

# 确保新数据的列名与训练数据一致
# 如果新数据包含不需要的列（如 '_GR', '_RT', '_CAL', '_RXO'），需要删除
new_data_cleaned = new_data.drop(['_GR', '_RT', '_CAL', '_RXO'], axis=1, errors='ignore')

# 使用加载的模型进行预测
predictions = loaded_model.predict(new_data_cleaned)

# 将预测结果添加到新数据中
new_data['预测孔隙度'] = predictions

# 输出预测结果
print("\n预测结果：")
print(new_data[['_GR', '_RT', '_AC', '_CAL', '_DEN', '_CNL', '_RXO', '预测孔隙度']])

# 可选：将预测结果保存到文件
new_data.to_csv('new_data_with_predictions.txt', sep='\t', index=False)
print("\n预测结果已保存到 'new_data_with_predictions.txt'")