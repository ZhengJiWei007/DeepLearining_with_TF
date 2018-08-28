'''
	DNN Drop Out
'''

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: dnn_dropout.py
@time: 2018/8/24 16:21
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
dropout_rate = 0.1  # == 1 - keep_prob

# 使用占位符节点表示 训练的 X,y
X = tf.placeholder(shape =(None,n_inputs),dtype = tf.float32,name='X')
y = tf.placeholder(shape=(None),dtype = tf.int64,name='y')
training = tf.placeholder_with_default(False,shape=(),name = 'training')

# 构建神经网络 (计算图)
with tf.name_scope('dnn') as scope:
	# he 权重初始化
	he_init = tf.contrib.layers.variance_scaling_initializer()

	# 利用 functools.partial 提高代码的重用性
	my_batch_norm_layer = partial(
		tf.layers.batch_normalization,
		momentum=batch_norm_momentum,
		training = training
	)

	my_dense_layer = partial(
		tf.layers.dense,
		kernel_initializer = he_init
	)
	'''
		dropout 防止过拟合
	'''
	my_dense_dropout = partial(
		tf.layers.dropout,
		rate = dropout_rate,
		training = training
	)
	# 对输入进行丢失
	X_dropout = my_dense_dropout(X)
	hidden1 = my_dense_layer(X_dropout, n_hidden1, name='hidden1')
	# 第一个隐层进行丢失
	hidden1_dropout = my_dense_dropout(hidden1)
	# 输入第一层 elu激活函数之前，先进行批量标准化
	bn1 = tf.nn.elu(my_batch_norm_layer(hidden1_dropout))
	# 输入第二层 elu激活函数
	hidden2 = my_dense_layer(bn1,n_hidden2 ,name = 'hidden2')
	# 对第二个隐层进行丢失
	hidden2_dropout = my_dense_dropout(hidden2)
	bn2 = tf.nn.elu(my_batch_norm_layer(hidden2_dropout))
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
from datetime import datetime

# tensorboard 可视化
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = r"D://tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
xentropy_summary = tf.summary.scalar('xentropy', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

mnist = input_data.read_data_sets("./data/")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 50
batch_size = 200
n_batches = mnist.train.num_examples // batch_size
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epoches): #梯度下降的迭代次数
		for batch_index in range(n_batches):
			X_batch,y_batch = mnist.train.next_batch(batch_size)
			if batch_index % 10 == 0 :
				xentropy_summary_str = xentropy_summary.eval(feed_dict={X: X_batch, y: y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(xentropy_summary_str,step)
			sess.run([training_op],
			         feed_dict={X:X_batch,y:y_batch,training:True})
		# 每次迭代完之后进行测试  观察精度是否提高
		acc = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
		#
		print(epoch, 'Test accuracy:', acc)
	save_path = saver.save(sess, "../model/my_model_final.ckpt")