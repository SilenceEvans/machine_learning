import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.数据处理
# 导入数据
iris = load_iris()

'''
# 构建pd文件
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

iris_df['Species'] = iris.target

def plot_iris(data, col1, col2):
    """
    用来绘制特征值跟类别之间的关系图
    :param data:
    :param col1:
    :param col2:
    """
    sns.lmplot(x=col1, y=col2, data=data, hue='Species', fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("鸢尾花种类分布")
    plt.show()

feature_names = iris.feature_names
plot_iris(iris_df, feature_names[0], feature_names[3])
'''


# 划分测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 2.特征工程
transformer = StandardScaler()
tr_x_train = transformer.fit_transform(x_train)
tr_x_test = transformer.transform(x_test)

# 3.模型训练
# 注意使用KNeighborsClassifier时，其中使用到的算法，KNeighborsClassifier会根据
# 数据集规模自行选择
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(tr_x_train, y_train)

# 4.进行预测
# 获取预测结果
pre_result = classifier.predict(tr_x_test)
print("预测结果为：\n", pre_result)
print("比对真实值和预测值：\n", pre_result == y_test)
# 计算预测准确率,如果是只获取准确率的话都不需要上一步的获取预测结果的代码
accuracy = classifier.score(tr_x_test, y_test)
print("准确率为：\n", accuracy)
