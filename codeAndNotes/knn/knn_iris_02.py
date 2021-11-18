import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 导入数据
iris = load_iris()
# 构建pd文件
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

iris_df['Species'] = iris.target


def plot_iris(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue='Species', fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("鸢尾花种类分布")
    plt.show()


feature_names = iris.feature_names
# plot_iris(iris_df, feature_names[0], feature_names[3])

# 划分测试集和训练集
x_train, y_train, x_test, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)
print("训练集特征值是：\n", x_train)
print("训练集目标值是：\n", x_train)

print("测试集特征值是：\n", x_test)
print("测试集特征值是：\n", y_test)
