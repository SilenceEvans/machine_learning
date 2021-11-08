import numpy as np
import pandas as pd

# 引入sklearn库中的数据集，iris(中文释义：鸢尾花)
from sklearn.datasets import load_iris
# 切分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
# 计算分类预测的准确率
from sklearn.metrics import accuracy_score
