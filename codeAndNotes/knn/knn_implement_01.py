import numpy as np
import pandas as pd

# 引入sklearn库中的数据集，iris(中文释义：鸢尾花)
from sklearn.datasets import load_iris
# 切分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
# 计算分类预测的准确率
from sklearn.metrics import accuracy_score

# 导入鸢尾花数据集
iris = load_iris()
# 用pandas分析该数据集
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 添加分类列
df['class'] = iris.target
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

# 构建特征矩阵和对应的分类结果的矩阵
x = iris.data
y = iris.target.reshape(-1, 1)  # 开奖分类结果构建成一个列向量，-1代表任意多的行
print(x.shape, y.shape)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)
"""
test_size 是指划分测试集的比例因子
random_state 相当于随机种子
stratify 指随机的时候主要依据哪个列随机，要使得y均匀的随机
"""


# 距离函数定义
# 1.欧氏距离
def euclidean_distance(a, b):
    # 逐行相减后，每行按行求和
    return np.sqrt(np.sum((a - b) ** 2, axis=1))  # axis = 0 是逐行相减后按列求和


# 2.曼哈顿距离
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


# 分类器的定义
class KNN:
    # 定义类变量
    x_train = []
    y_train = []

    # 定义构造方法,给变量赋初始值之后，才可以在创建对象的时候不用传参数
    def __init__(self, neighbors=1, dis_func=euclidean_distance):
        """
        neighbors:选要分类的点周围那些点为它的邻点
        dis-func:分类器要使用的距离函数
        """
        self.neighbors = neighbors
        self.dis_func = dis_func

    # 定义模型训练方法
    def fit_model(self, x, y):
        """
        定义模型训练需要的参数
        """
        self.x_train = x
        self.y_train = y

    # 定义模型预测方法
    def model_pred(self, pred_x):
        """
        pred_x:需要预测的特征值
        预测完的y跟pred_x是行数相同的列向量
        生成一个跟pred_x同行数的零向量，用来存储每个数据的预测结果
        """
        y_pred = np.zeros((pred_x.shape[0], 1), dtype=self.y_train.dtype)

        # 遍历需要预测的x矩阵的特征值，输入到距离函数里计算距离
        # 拿到枚举的索引和值
        for i, every_pred_x in enumerate(pred_x):
            # 计算某点到其余所有点的距离
            distance_list = self.dis_func(self.x_train, every_pred_x)
            # 排序，选取最近的k个点
            all_disdance_index = np.argsort(distance_list)  # 这个方法返回的是排序后每个值及之前的索引值
            # 先取出此时前k个值，切片
            neighbors_index = all_disdance_index[:self.neighbors]
            # 再根据这些索引取出训练集中对应的类别
            results_pred_y = self.y_train[neighbors_index].ravel()
            # 统计类别中出现频率最高的那个，赋值给y_pred存储
            # bincount()函数其统计的是一个数组中的数出现的频率，有意思的是排序后的值，索引代表的是原数组中的那些值，值代表的是出现的频率
            # argmax()函数获取的是出现最多这个值得索引
            frequency_list = np.bincount(results_pred_y)
            max_fre_y = np.argmax(frequency_list)
            y_pred[i] = max_fre_y

        return y_pred


# 对比不同的距离方法，不同的邻近点数量的准确率
knn = KNN()
knn.fit_model(x_train, y_train)
result_list = []
# 考虑不同的距离
for i in [0, 1]:
    knn.dis_func = euclidean_distance if i == 0 else manhattan_distance

    # 不同邻近点下的预测准确率,从1到10，步长为2
    for k in range(1, 10, 2):
        knn.neighbors = k
        y_pred = knn.model_pred(x_test)
        # 求出预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'euclidean_distance' if i == 0 else 'manhattan_distance', accuracy])

result_data = pd.DataFrame(result_list, columns=['k', 'function', 'accuracy'])
