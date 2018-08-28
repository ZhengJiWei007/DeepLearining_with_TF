'''
	8.深度学习。
		i.建立一个 DNN，有五个隐藏层，每层 100 个神经元，使用 He 初始化和 ELU 激活函数。
		ii.使用 Adam 优化和提前停止，请尝试在 MNIST 上进行训练，但只能使用数字 0 到 4，因为我们将在下一个练习中在数字 5 到 9 上进行迁移学习。 您需要一个包含五个神经元的 softmax 输出层，并且一如既往地确保定期保存检查点，并保存最终模型，以便稍后再使用它。
		iii.使用交叉验证调整超参数，并查看你能达到什么准确度。
		iv.现在尝试添加批量标准化并比较学习曲线：它是否比以前收敛得更快？ 它是否会产生更好的模型？
		v.模型是否过拟合训练集？ 尝试将 dropout 添加到每一层，然后重试。 它有帮助吗？
'''
"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: test_8.py
@time: 2018/8/27 10:31
"""

import tensorflow as tf



n_input = 28 * 28
n_hidden1 = 100
n_hidden2 = 100
n_hidden3 = 100
n_hidden4 = 100
n_hidden5 = 100
n_output =5

'''
	构建网络图
'''

X = tf.placeholder(shape=(None,n_input),dtype=tf.float32,name='X')
y = tf.placeholder(shape=(None),dtype=tf.int64,name='y')


with tf.name_scope('dnn'):
	# he初始化
	tf.contrib.layers.variance_scaling_initializer()
	hidden1 = tf.layers.dense(X, n_input, name='hidden1', activation=tf.nn.elu)
	hidden2 = tf.layers.dense(hidden1,n_hidden1,activation=tf.nn.elu)















