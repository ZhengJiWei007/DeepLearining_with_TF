#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: manual_cal_gradient.py
@time: 2018/7/2 9:09
"""


# 手动计算梯度
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# 加载数据
housing = fetch_california_housing()
m , n = housing.data.shape
n_epochs = 1000
learning_rate = 0.01
# 增加偏置项
housing_plus_bias = np.c_[np.ones((m,1)),housing.data]
# 数据标准化（梯度下降更快）
housing_plus_bias_scaled = StandardScaler().fit_transform(housing_plus_bias)
X = tf.constant(housing_plus_bias_scaled,dtype=np.float32,name='X')
y = tf.constant(housing.target.reshape(-1,1),dtype=np.float32,name='y')
# 随机初始化theta
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
y_pred = tf.matmul(X,theta,name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name='mse')
# 计算梯度 m/2 * XT * error
gradients = 2/m * tf.matmul(tf.transpose(X),error)
# 更新 theta = theta - learning_rate * gradients
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch", epoch, "MSE =", mse.eval())
		sess.run(training_op) # 随机梯度下降
	best_theta = theta.eval()
	print(best_theta)