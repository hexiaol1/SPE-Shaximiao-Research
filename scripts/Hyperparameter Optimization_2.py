import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. 读取数据
data = pd.read_csv("D:\\repository\\spe\\data\\XGBoost_sample.txt", sep="\t")
X = data.drop(columns=["标签"])
y = data["标签"] - 1  # 修正标签从0开始

# 过滤掉不需要的特征列
drop_columns = ["相对重心", "变差方差根", "平均中位数", "平均斜率"]
X = X.drop(columns=[col for col in drop_columns if col in X.columns])

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 定义XGBoost模型
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

# 5. 定义超参数搜索空间
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

# 6. 网格搜索 + 5折交叉验证
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# 7. 开始训练
grid_search.fit(X_train, y_train)

# 8. 输出最优参数和训练集验证准确率
print("最佳参数：", grid_search.best_params_)
print("交叉验证最佳准确率：", grid_search.best_score_)

# 9. 测试集准确率
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率：{test_accuracy:.4f}")
