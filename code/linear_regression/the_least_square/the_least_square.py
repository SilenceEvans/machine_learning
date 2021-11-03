import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt('data.csv', delimiter=',')
# delimiter 分隔符
# 提出points中的第一列和第二列
x = points[:, 0]
y = points[:, 1]
# 绘制所有点，scatter v. 撒，播撒；（使）散开，（使）散布在各处；（物理）散射（电磁辐射或粒子）；（棒球）（被击中但没有得分的）有效投（球）
#  n. 零星散布的东西；（统计）（对某一变量作反复测量或观察所得数值的）离差；（物理）（光、其他电磁波或粒子的）散射
plt.scatter(x, y)
plt.show()


# 2.定义拟合函数

def compute_loss(a, b, points):
    all_loss = 0
    length = len(points)
    for item in range(length):
        x = points[item, 0]
        y = points[item, 1]
        all_loss += (y - a * x - b) ** 2
    return all_loss / length


# 3.定义拟合函数


# 先定义一个计算平均值的函数
# points:点数组，indicator：元素索引
def average(points, indicator):
    length = len(points)
    sum = 0
    for item in range(length):
        item = points[item, indicator]
        sum += item

    ave = sum / length
    return ave


# 先定义计算a的函数
def calculate_a(points):
    # 定义a
    a = 0
    # 定义一个变量接收分子的和
    sum0 = 0
    length = len(points)
    ave = average(points, 0)
    for item in range(length):
        sum0 += points[item, 1] * (points[item, 0] - ave)

    # 定义三个变量接受分母的值
    # 分母第一项，每个x平方项累加之和
    sum1 = 0
    for item in range(length):
        sum1 += (points[item, 0] ** 2)
    # 分母第二项
    sum2 = 0
    sum2_square = 0
    sec2_value = 0
    for item in range(length):
        sum2 += (points[item, 0])

    sec2_value = (sum2 ** 2) / length

    # 分母总值
    sum3 = 0
    sum3 = sum1 - sec2_value

    # 计算a的值
    a = sum0 / sum3

    return a


# 定义计算b的函数
def calculate_b(points, a):
    b = 0
    length = len(points)
    sum_b = 0
    for item in range(length):
        sum_b += (points[item, 1] - a * points[item, 0])

    b = sum_b / length
    return b


# 4.test
# 获取a,b
a = calculate_a(points)
b = calculate_b(points, a)
print(a, b)
# 获取总的损失
all_loss = compute_loss(a, b, points)
print(all_loss)

plt.scatter(x, y)
pred_y = a * x + b
plt.plot(x, pred_y, c='r')
plt.show()
