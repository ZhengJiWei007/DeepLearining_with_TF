
# 使用tensorFlow计算线性回归参数 theta

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

pd.set_option('display.max_columns',15)
np.set_printoptions(suppress=True) #不显示科学技术发
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
# plt.rcParams['axes.unicode_minus'] = False

housing = fetch_california_housing()
m, n = housing.data.shape
# 增加偏置项
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias,dtype=np.float32,name='X')
y = tf.constant(housing.target.reshape(-1,1),dtype=np.float32,name='y')
XT = tf.transpose(X)

# (X.T * X)-1 * X.T * y
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
	theta_value = theta.eval()
print(theta_value)



















