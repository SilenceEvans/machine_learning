# 导包
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error

# 加载数据
housing = fetch_california_housing()
# 分割数据集
train_x, test_x, train_y, test_y = train_test_split(housing.data, housing.target, test_size=0.2, random_state=0)

# 特征工程
# 标准化
# 定义转换器
transfer = StandardScaler()
train_x = transfer.fit_transform(train_x)
test_x = transfer.fit_transform(test_x)


# 模型训练
def linear1(train_x, train_y):
    '''
    使用正规方程进行线性回归
     Args:
        train_x:
        train_y:

    Returns:

    '''
    estimator = LinearRegression()
    estimator.fit(train_x, train_y)
    return estimator


def linear2(train_x, train_y):
    '''
    使用梯度下降进行线性回归
    Args:
        train_x:
        train_y:

    Returns:

    '''
    estimaor = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.0001)
    estimaor.fit(train_x, train_y)
    return estimaor


'''
要注意的是：使用梯度下降算法是需要不断的调整学习率的大小去获得最优解的，在案例中老师用的是0。01，但因为我采用的是新数据集
所以，我将学习率一直调整到0.0001时才获得了和正规方程差不多的误差值
'''
# 模型评估
estimator_0 = linear1(train_x, train_y)
estimator_1 = linear2(train_x, train_y)
pre_y0 = estimator_0.predict(test_x)
pre_y1 = estimator_1.predict(test_x)
# 0.获取系数
print(estimator_0.coef_, estimator_1.coef_)
# 1.获取偏置
print(estimator_0.intercept_, estimator_1.intercept_)
# 2.获取误差
err0 = mean_squared_error(test_y, pre_y0)
err1 = mean_squared_error(test_y, pre_y1)
print(err0, err1)
