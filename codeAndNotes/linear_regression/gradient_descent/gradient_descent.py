import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt('data.csv', delimiter=',')
# delimiter 分隔符
# obtain the first column and second column in the array of points
x = points[:, 0]
y = points[:, 1]
# 绘制所有点，scatter v. 撒，播撒；（使）散开，（使）散布在各处；（物理）散射（电磁辐射或粒子）；（棒球）（被击中但没有得分的）有效投（球）
#  n. 零星散布的东西；（统计）（对某一变量作反复测量或观察所得数值的）离差；（物理）（光、其他电磁波或粒子的）散射
plt.scatter(x, y)
plt.show()


# 2.Define the function to compute all cost
def compute_loss(theta0, theta1, points):
    '''
    Args:
        theta0: the coefficient of unknown (like X0,X1...Xn)
        theta1: the constant term
        points:

    Returns: the value of cost
    '''
    all_loss = 0
    length = len(points)
    for item in range(length):
        x = points[item, 0]
        y = points[item, 1]
        all_loss += (y - theta0 * x - theta1) ** 2
    return all_loss / length


# 3.implement the gradient descent
# Firstly,define a function to  compute the gradient according to the current theta0 and theta1
def compute_gradient(cur_theta0, cur_theta1, points):
    '''
    Args:
        cur_theta0: the current theta0
        cur_theta1: the current theta1
        points: the set of points

    Returns: the current gradient

    '''
    length = len(points)
    sum_theta0 = 0
    sum_theta1 = 0
    for item in range(length):
        sum_theta0 += (cur_theta0 * points[item, 0] + cur_theta1 - points[item, 1]) * points[item, 0]
        sum_theta1 += (cur_theta0 * points[item, 0] + cur_theta1 - points[item, 1])

    partial_the0 = 2 / length * sum_theta0
    partial_the1 = 2 / length * sum_theta1
    return partial_the0, partial_the1


# define the function to compute the updated gradient in the next step
def gradient_descent(cur_theta0, cur_theta1, alpha, iter_num, points):
    '''
    To compute the updated gradient
    Args:
        cur_theta0:  current theta0
        cur_theta1: current theta1
        alpha: step value
        iter_num: the number of iteration
        points: set of points

    Returns:
        updated_the0:
        updated_the1:
        cost_list: the array of cost,index is iteration's order
    '''
    cost_list = []
    for item in range(iter_num):
        cost = compute_loss(cur_theta0, cur_theta1, points)
        cost_list.append(cost)
        # obtain the current gradient
        partial_th0, partial_th1 = compute_gradient(cur_theta0, cur_theta1, points)
        updated_the0 = cur_theta0 - (alpha * partial_th0)
        updated_the1 = cur_theta1 - (alpha * partial_th1)
        cur_theta0 = updated_the0
        cur_theta1 = updated_the1
    return updated_the0, updated_the1, cost_list


# 4.Test
# Output the result when variable equals the value given below
theta0 = 0
theta1 = 0
alpha = 0.0001
iter_num = 10
theta0, theta1, cost_list = gradient_descent(theta0, theta1, alpha, iter_num, points)
all_loss = compute_loss(theta0, theta1, points)
plt.plot(cost_list)
plt.show()
print( theta0, theta1, all_loss)

# plot the image of fitting function
plt.scatter(x, y)
pred_y = theta0 * x + theta1
plt.plot(x, pred_y, c='r')
plt.show()
