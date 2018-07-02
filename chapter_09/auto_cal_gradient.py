#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: auto_cal_gradient.py
@time: 2018/7/2 10:11
"""


#使用tensorflow自动计算梯度

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# 加载数据
housing = fetch_california_housing()
m , n = housing.data.shape
n_epochs = 10
learning_rate = 0.01 #可调
# 增加偏置项
housing_plus_bias = np.c_[np.ones((m,1)),housing.data]
# 数据标准化（梯度下降更快）
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.placeholder(shape=(None,n+1),name='X',dtype=np.float32)
y = tf.placeholder(shape=(None,1),name='y',dtype=np.float32)

# 随机初始化 theta
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0))
y_pred = tf.matmul(X,theta,name='predictions')

# 名称作用域
with tf.name_scope('loss') as scope:
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error),name='mse')


# 自动计算梯度
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
batch_size = 100
n_batches = int(np.ceil(m/batch_size))
# 使用tensorboard 可视化 训练 阶段 mse
now = datetime.now().strftime("%Y%m%d%H%M%S")
root_logdir = r"D://tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


def fetch_batch(epoch, batch_index, batch_size):
    know = np.random.seed(epoch * n_batches + batch_index)  # not shown in the book  
    # print("我是know:",know)
    indices = np.random.randint(m, size=batch_size)  # not shown  
    X_batch = scaled_housing_data_plus_bias[indices] # not shown  
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown  
    return X_batch, y_batch  


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch,y_batch = fetch_batch(epoch, batch_index, batch_size)
			if batch_index %  10 == 0:
				summary_str = mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str,step)
			sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
	best_theta = theta.eval()
		# if epoch % 100 == 0:
		# 	print("Epoch", epoch, "MSE =", mse.eval())  # not shown
		# 	save_path = saver.save(sess, "./model/my_model.ckpt")
file_writer.flush()
file_writer.close()
print(best_theta)




