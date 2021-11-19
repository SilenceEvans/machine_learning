from sklearn.neighbors import KNeighborsClassifier

# 1.构造数据
x = [[1], [2], [10], [20]]
y = [0, 0, 1, 1]

# 实例化估计器对象
classifier = KNeighborsClassifier(n_neighbors=1)

# 训练模型
classifier.fit(x, y)

# 数据预测
cla = classifier.predict([[0]])

print(cla)
