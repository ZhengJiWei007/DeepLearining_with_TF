'''
	DNN 防止过拟合的办法 L1,L2 norm
'''

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: dnn_regularizer.py
@time: 2018/8/24 15:44
"""


import tensorflow as tf
from functools import partial

'''
	构造阶段
'''

n_inputs = 28 * 28 # 输入神经元的个数
n_hidden1 = 300
n_hidden2 = 100
n_outputs  = 10 #输出神经元的个数


batch_norm_momentum = 0.9 #批量标准化的缩放参数 y = alpha * Z + beta
learning_rate = 0.01 #梯度下降的学习率

# 使用占位符节点表示 训练的 X,y
X = tf.placeholder(shape =(None,n_inputs),dtype = tf.float32,name='X')
y = tf.placeholder(shape=(None),dtype = tf.int64,name='y')
training = tf.placeholder_with_default(False,shape=(),name = 'training')

# 构建神经网络 (计算图)
with tf.name_scope('dnn') as scope:
	# he 权重初始化
	he_init = tf.contrib.layers.variance_scaling_initializer()
	l1_reg = tf.contrib.layers.l1_regularizer(0.01)
	# 利用 functools.partial 提高代码的重用性
	my_batch_norm_layer = partial(
		tf.layers.batch_normalization,
		momentum=batch_norm_momentum,
		training = training
	)
	'''
		方法一 ：添加 L1 , L2正则项
	'''
	my_dense_layer = partial(
		tf.layers.dense,
		kernel_initializer = he_init,
		kernel_regularizer = l1_reg
	)

	hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
	# 输入第一层 elu激活函数之前，先进行批量标准化
	bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
	# 输入第二层 elu激活函数
	hidden2 = my_dense_layer(bn1,n_hidden2 ,name = 'hidden2')
	bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
	# 输入到输出层
	logists_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
	logists = my_batch_norm_layer(logists_before_bn)

# 定义损失
with tf.name_scope('loss') :
	# 交叉熵
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logists) # 这将给我们一个包含每个实例的交叉熵的 1D 张量
	# 计算所有实例的平均交叉熵
	base_loss = tf.reduce_mean(xentropy,name='avg_xentropy')
	reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	# 把 正则项 损失加到整体损失上
	loss = tf.add_n([base_loss] + reg_loss ,name = 'loss')

# 梯度下降求解参数
with tf.name_scope('training') :
	# 运行 batch_normalization 额外需要的更新操作
	extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# 自动求解
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	with tf.control_dependencies(extra_update_ops):
		training_op = optimizer.minimize(loss=loss) # minimize 负责计算梯度

	# 应用梯度裁剪 (在递归神经网络中 非常有用)
	# threshhold = 1
	# grads_and_vars = optimizer.compute_gradients(loss=loss)
	# capped_gvs = [(tf.clip_by_value(grad,-threshhold,threshhold),var)for grad,var in grads_and_vars]
	# training_op = optimizer.apply_gradients(capped_gvs)
	#

# 评估模型 精度
with tf.name_scope('eval') :
	correct = tf.nn.in_top_k(predictions=logists, targets=y, k=1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

'''
	执行阶段
'''
# 加载数据
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 将数据随机划分成 若干个batch
def shuffle_batch(X,y,batch_size):
	rnd_idx = np.random.permutation(len(X)) # 将list的id 随机打乱生成 list
	n_batches = len(X) // batch_size
	for batch_idx in np.array_split(rnd_idx,n_batches):
		X_batch , y_batch = X[batch_idx] , y[batch_idx]
		yield X_batch , y_batch


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
mnist = input_data.read_data_sets("./data/")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 50
batch_size = 200

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epoches): #梯度下降的迭代次数
		for X_batch,y_batch in shuffle_batch(X_train,y_train,200):
			sess.run([training_op],
			         feed_dict={X:X_batch,y:y_batch,training:True})
		# 每次迭代完之后进行测试  观察精度是否提高
		acc = accuracy.eval(feed_dict={X:X_test,y:y_test})
		print(epoch, 'Test accuracy:', acc)