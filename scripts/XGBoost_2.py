import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. 读取txt数据
data = pd.read_csv("D:\\repository\\spe\\data\\XGBoost_sample.txt", sep="\t")

# 2. 特征与标签
drop_cols = ["标签", "相对重心", "变差方差根", "平均中位数", "平均斜率"]
X = data.drop(columns=drop_cols)
y = data["标签"]

# 🚨 修正标签范围：1-7 → 0-6
y = y - 1

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 模型
model = XGBClassifier(
    learning_rate=0.2,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=1,
    random_state=42
)
model.fit(X_train, y_train)

# 6. 预测与准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率：{accuracy:.4f}")
