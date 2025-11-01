import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class GeologicalFeatureCalculator:
    def __init__(self, layer_file, gr_file):
        """
        初始化地质特征计算器

        Parameters:
        - layer_file: 分层数据文件路径
        - gr_file: GR测井数据文件路径
        """
        self.layer_file = layer_file
        self.gr_file = gr_file
        self.layers_df = None
        self.gr_df = None
        self.features_df = None

    def load_data(self):
        """加载分层数据和GR数据"""
        # 加载分层数据
        self.layers_df = pd.read_csv(self.layer_file, sep='\t')
        print(f"分层数据加载完成，共 {len(self.layers_df)} 层")

        # 加载GR数据
        self.gr_df = pd.read_csv(self.gr_file, sep='\t')
        print(f"GR数据加载完成，共 {len(self.gr_df)} 个数据点")

        # 数据预览
        print("\n分层数据前5行:")
        print(self.layers_df.head())
        print("\nGR数据前5行:")
        print(self.gr_df.head())

    def calculate_relative_center_of_gravity(self, gr_values, depths):
        """
        计算GR曲线的相对重心

        Parameters:
        - gr_values: GR值数组
        - depths: 对应的深度数组

        Returns:
        - 相对重心 (0-1之间的值)
        """
        if len(gr_values) == 0:
            return np.nan

        # 计算加权平均深度
        weighted_depth = np.sum(gr_values * depths) / np.sum(gr_values)

        # 计算深度范围
        min_depth = np.min(depths)
        max_depth = np.max(depths)

        # 计算相对重心 (归一化到0-1)
        relative_cog = (weighted_depth - min_depth) / (max_depth - min_depth)

        return relative_cog

    def calculate_variation_variance_root(self, gr_values):
        """
        计算变差方差根 (变异函数的平方根)

        Parameters:
        - gr_values: GR值数组

        Returns:
        - 变差方差根
        """
        if len(gr_values) < 2:
            return np.nan

        # 计算变异函数 (半方差)
        n = len(gr_values)
        variances = []

        for h in range(1, min(10, n)):  # 考虑前10个滞后距离
            gamma_h = 0
            count = 0
            for i in range(n - h):
                gamma_h += (gr_values[i + h] - gr_values[i]) ** 2
                count += 1
            if count > 0:
                variances.append(gamma_h / (2 * count))

        # 返回变差方差根 (变异函数的平方根)
        if variances:
            return np.sqrt(np.mean(variances))
        else:
            return np.nan

    def calculate_mean_median(self, gr_values):
        """
        计算平均中位数 (平均值和中位数的平均值)

        Parameters:
        - gr_values: GR值数组

        Returns:
        - 平均中位数
        """
        if len(gr_values) == 0:
            return np.nan

        mean_val = np.mean(gr_values)
        median_val = np.median(gr_values)

        return (mean_val + median_val) / 2

    def calculate_mean_slope(self, gr_values, depths):
        """
        计算平均斜率

        Parameters:
        - gr_values: GR值数组
        - depths: 对应的深度数组

        Returns:
        - 平均斜率
        """
        if len(gr_values) < 2:
            return np.nan

        slopes = []
        for i in range(len(gr_values) - 1):
            depth_diff = depths[i + 1] - depths[i]
            gr_diff = gr_values[i + 1] - gr_values[i]

            if depth_diff != 0:
                slope = gr_diff / depth_diff
                slopes.append(slope)

        if slopes:
            return np.mean(np.abs(slopes))  # 使用绝对值的平均
        else:
            return np.nan

    def calculate_features_for_all_layers(self):
        """为所有分层计算特征"""
        features_list = []

        for idx, layer in self.layers_df.iterrows():
            top = layer['Top']
            bottom = layer['Bottom']
            text_label = layer['Text']

            # 筛选该层内的GR数据
            layer_gr_data = self.gr_df[
                (self.gr_df['Depth'] >= top) &
                (self.gr_df['Depth'] <= bottom)
                ]

            if len(layer_gr_data) < 2:
                print(f"警告: 第 {idx + 1} 层 ({top}-{bottom}) 数据点不足，跳过")
                continue

            depths = layer_gr_data['Depth'].values
            gr_values = layer_gr_data['GR2'].values

            # 计算各个特征
            relative_cog = self.calculate_relative_center_of_gravity(gr_values, depths)
            variation_variance_root = self.calculate_variation_variance_root(gr_values)
            mean_median = self.calculate_mean_median(gr_values)
            mean_slope = self.calculate_mean_slope(gr_values, depths)

            # 存储结果
            features_list.append({
                'Layer_Index': idx + 1,
                'Top': top,
                'Bottom': bottom,
                'Thickness': bottom - top,
                'Text_Label': text_label,
                'Data_Points': len(layer_gr_data),
                'Relative_Center_of_Gravity': relative_cog,
                'Variation_Variance_Root': variation_variance_root,
                'Mean_Median': mean_median,
                'Mean_Slope': mean_slope,
                'GR_Mean': np.mean(gr_values),
                'GR_Std': np.std(gr_values)
            })

        self.features_df = pd.DataFrame(features_list)
        return self.features_df

    def visualize_results(self):
        """可视化结果"""
        if self.features_df is None or len(self.features_df) == 0:
            print("没有可可视化的数据")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('地质特征分析结果', fontsize=16)

        # 相对重心分布
        axes[0, 0].bar(self.features_df['Layer_Index'],
                       self.features_df['Relative_Center_of_Gravity'])
        axes[0, 0].set_title('相对重心')
        axes[0, 0].set_xlabel('层序号')
        axes[0, 0].set_ylabel('相对重心')

        # 变差方差根分布
        axes[0, 1].bar(self.features_df['Layer_Index'],
                       self.features_df['Variation_Variance_Root'])
        axes[0, 1].set_title('变差方差根')
        axes[0, 1].set_xlabel('层序号')
        axes[0, 1].set_ylabel('变差方差根')

        # 平均中位数分布
        axes[0, 2].bar(self.features_df['Layer_Index'],
                       self.features_df['Mean_Median'])
        axes[0, 2].set_title('平均中位数')
        axes[0, 2].set_xlabel('层序号')
        axes[0, 2].set_ylabel('平均中位数')

        # 平均斜率分布
        axes[1, 0].bar(self.features_df['Layer_Index'],
                       self.features_df['Mean_Slope'])
        axes[1, 0].set_title('平均斜率')
        axes[1, 0].set_xlabel('层序号')
        axes[1, 0].set_ylabel('平均斜率')

        # GR均值分布
        axes[1, 1].bar(self.features_df['Layer_Index'],
                       self.features_df['GR_Mean'])
        axes[1, 1].set_title('GR均值')
        axes[1, 1].set_xlabel('层序号')
        axes[1, 1].set_ylabel('GR均值')

        # 厚度分布
        axes[1, 2].bar(self.features_df['Layer_Index'],
                       self.features_df['Thickness'])
        axes[1, 2].set_title('层厚度')
        axes[1, 2].set_xlabel('层序号')
        axes[1, 2].set_ylabel('厚度 (m)')

        plt.tight_layout()
        plt.show()

    def save_results(self, output_file='geological_features_results.csv'):
        """保存结果到CSV文件"""
        if self.features_df is not None:
            self.features_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"结果已保存到: {output_file}")
        else:
            print("没有结果可保存")


# 使用示例
if __name__ == "__main__":
    # 初始化计算器
    calculator = GeologicalFeatureCalculator(
        layer_file='D:\\repository\spe\data\Well stratification.txt',  # 替换为你的分层数据文件路径
        gr_file='D:\\repository\spe\data\Single-well GR.txt'  # 替换为你的GR数据文件路径
    )

    # 加载数据
    calculator.load_data()

    # 计算特征
    features = calculator.calculate_features_for_all_layers()

    # 显示结果
    print("\n计算完成！特征结果:")
    print(features)

    # 可视化结果
    calculator.visualize_results()

    # 保存结果
    calculator.save_results('D:\\repository\spe\data\Morphological parameters.csv')

    # 输出统计摘要
    print("\n特征统计摘要:")
    if features is not None and len(features) > 0:
        numeric_features = features.select_dtypes(include=[np.number])
        print(numeric_features.describe())