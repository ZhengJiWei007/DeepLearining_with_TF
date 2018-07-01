
'''
	第一个神经网络的计算
'''

import tensorflow as tf

# 创建计算图
x = tf.Variable(initial_value=3,name='x')
y = tf.Variable(initial_value=4,name='y')
f = x * x * y + y +2

# 第一种写法
# session = tf.Session()
# session.run(x.initializer)
# session.run(y.initializer)
# result = session.run(f)
# print(result)
# session.close()

# 第二种简洁语法
with tf.Session() as session:
	x.initializer.run() # 手动初始化变量
	y.initializer.run()
	result = f.eval()
print(result)


# 第三种使用全局变量初始化，而不是手动初始化每个变量
initializer = tf.global_variables_initializer()
with tf.Session() as session:
	initializer.run()
	result = f.eval()
print(result)




















