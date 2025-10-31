import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('孔隙度+标签.txt', sep='\t', encoding='utf-8')


# 定义渗透率计算公式
def permeability_by_class(label, porosity):
    formulas = {
        1: lambda phi: 0.0011 * np.exp(0.4046 * phi),
        2: lambda phi: 0.0025 * np.exp(0.3717 * phi),
        3: lambda phi: 0.0078 * np.exp(0.3255 * phi),
        4: lambda phi: 0.0046 * np.exp(0.4625 * phi),
        5: lambda phi: 0.0131 * np.exp(0.4120 * phi),
        6: lambda phi: 0.0453 * np.exp(0.3429 * phi),
        7: lambda phi: 0.0127 * np.exp(0.6501 * phi),
        'Unclassified': lambda phi: 0.0221 * np.exp(0.3458 * phi)
    }

    return formulas.get(label, formulas['Unclassified'])(porosity)


def unified_permeability(porosity):
    return 0.0221 * np.exp(0.3458 * porosity)


# 计算按标签分开的渗透率
data['Permeability_by_class'] = data.apply(lambda row: permeability_by_class(row['标签'], row['POR']), axis=1)

# 计算统一公式渗透率
data['Permeability_unified'] = data['POR'].apply(unified_permeability)

# 保存结果到 CSV 文件
data[['深度', 'Permeability_by_class']].to_csv('permeability_by_class.csv', index=False, encoding='utf-8')
data[['深度', 'Permeability_unified']].to_csv('permeability_unified.csv', index=False, encoding='utf-8')

print("渗透率预测结果已保存为 'permeability_by_class.csv' 和 'permeability_unified.csv'.")
