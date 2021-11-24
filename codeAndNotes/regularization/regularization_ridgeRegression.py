from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.metrics import mean_squared_error

# 加载数据
housing = fetch_california_housing()
# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=2, random_state=20)
# 特征工程
# 对数据集进行标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 训练模型
# 正则化稀疏选1
# estimator = Ridge(alpha=1)
estimator = RidgeCV()
estimator.fit(x_train, y_train)

# 模型预测
y_predict = estimator.predict(x_test)
print("系数是\n", estimator.coef_)
print("偏置是\n", estimator.intercept_)
print("正则化系数是\n",estimator.alpha_)
# 获取误差
err = mean_squared_error(y_test, y_predict)
print("误差是\n", err)
