# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:02:35 2024

@author: Dell
"""
import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import mode

# ------------------------ 自定义集成类 ------------------------ #
class HardVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        return self

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        mode_result = mode(preds, axis=0, keepdims=False)
        return mode_result.mode.flatten()

data = pd.read_csv(r'1砂组数据.txt', delimiter='\t', encoding='utf-8')
#data = pd.read_csv('参数结果.txt', delimiter='\t', encoding='utf-8')
# 提取特征和标签
X = data.drop(columns=[ '深度'])  # 特征矩阵，移除了深度列
#X = data.drop(columns=['深度',  '平均中位数',  '变差方差根',  '相对重心',  '平均斜率'])  # 特征矩阵，移除了深度列



# 加载模型
loaded_rf_model = load('best_hard_voting_model.pkl')
#loaded_rf_model = load('random_forest_model.joblib2')
# 使用加载的模型进行预测
y_pred_loaded = loaded_rf_model.predict(X)

# 评估加载的模型性能
#accuracy_loaded = accuracy_score(y, y_pred_loaded)
#print("Loaded model accuracy:", accuracy_loaded)

# 创建包含预测结果的DataFrame
prediction_df = pd.DataFrame({'预测标签': y_pred_loaded})
# Modify prediction_df to include columns '顶深' and '底深'
prediction_df['深度'] = data['深度']


# 将DataFrame写入CSV文件
prediction_df.to_csv('离散结果.csv', index=False)



