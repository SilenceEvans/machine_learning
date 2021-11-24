from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

# 加载数据
california = fetch_california_housing()
# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=20)
# 特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 模型训练
# estimator = Ridge()
# estimator.fit(x_train, y_train)

# 保存训练的模型,注意保存的文件名要为'.pkl'
# joblib.dump(estimator, './demo.pkl')

# 加载训练的模型
estimator = joblib.load('./demo.pkl')

# 模型预测
y_predict = estimator.predict(x_test)
print(y_predict)
