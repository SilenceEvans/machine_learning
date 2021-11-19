# 使用交叉验证（Cross Validation）和网格搜索(Grid Research)来对模型训练进行优化
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.数据处理
# 导入数据
iris = load_iris()

# 划分测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 2.特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 3.模型训练与调优——交叉验证与网格搜索
# 注意使用KNeighborsClassifier时，其中使用到的算法，KNeighborsClassifier会根据数据集规模自行选择
# 要使用的超参数,使用5折交叉验证
estimator = KNeighborsClassifier()
param_dict = {'n_neighbors': [1, 3, 5, 7]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)
estimator.fit(x_train, y_train)

# 4.进行预测
# 获取预测结果
pre_result = estimator.predict(x_test)
print("预测结果为：\n", pre_result)
print("比对真实值和预测值：\n", pre_result == y_test)
# 计算预测准确率,如果是只获取准确率的话都不需要上一步的获取预测结果的代码
accuracy = estimator.score(x_test, y_test)
print("准确率为：\n", accuracy)

print("交叉验证中最好的结果：\n", estimator.best_score_)
print("交叉验证中最好的参数模型：\n", estimator.best_estimator_.__dict__)
print("每次交叉验证的结果：\n", estimator.cv_results_)
