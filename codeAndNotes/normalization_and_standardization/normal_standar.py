from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 归一化实现（implement the normalization）
data_df = pd.read_csv('./data/dating.txt')

# 将特征值归一到2～3之间
# 1.实现一个转换器
transformer = MinMaxScaler(feature_range=(2, 3))
# 调用fit_transform,注意其中参数数组是二维数组
data_df0 = transformer.fit_transform(data_df[['milage', 'Liters', 'Consumtime']])
print("最小值最大值归一化后的值为：\n", data_df0)
print("归一之前的值为\n", data_df)

# 标准化的实现(implement the standardization)
transformer_st = StandardScaler()
data_df1 = transformer_st.fit_transform(data_df[['milage', 'Liters', 'Consumtime']])
print("标准化之后的值为\n", data_df1)
