
'''
	使用TensorFlow 普通API 训练DNN 识别手写数字
'''
from time import time
import numpy as np
import tensorflow as tf


'''
	构造阶段
'''
n_inputs = 28 * 28  # 输入神经元的数量 - 列
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# 使用占位符节点表示训练数据个目标

X = tf.placeholder(shape=(None,n_inputs),dtype=tf.float32,name='X')
y = tf.placeholder(shape=(None),dtype=tf.int64,name='y')

# 构建神经网络（计算图）
with tf.name_scope('dnn') as scope:
	hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')
	hidden2 = tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.relu,name='hidden2')
	logits = tf.layers.dense(hidden2, n_outputs, activation=tf.nn.relu, name='outputs') # 通过softmax函数之前的网络输出

# 定义损失
with tf.name_scope('loss') as scope:
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # 这将给我们一个包含每个实例的交叉熵的 1D 张量
	# 计算所有实例的平均交叉熵
	loss = tf.reduce_mean(xentropy,name='loss')

# 梯度下降
learing_rate = 0.01
with tf.name_scope('train') as scope:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate)
	training_op = optimizer.minimize(loss)

# 评估模型 精度
with tf.name_scope('eval') as scope:
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



'''
	执行阶段
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 加载数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/")

n_epochs = 10000
batch_size = 50
t1 = time()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(mnist.train.num_examples // batch_size):
			X_batch,y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
		# 评估最后一个小批量和完整测试集上的模型
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
	save = saver.save(sess, save_path='./my_model_final.ckpt')
print('time:',(time()-t1)/60)
















