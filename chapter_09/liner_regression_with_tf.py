
# 使用tensorFlow计算线性回归参数 theta

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

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



















