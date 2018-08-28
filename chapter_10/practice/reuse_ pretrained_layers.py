'''  重用tf
    情况：重用 第一个隐层 ，重新定义弟二个 隐层和输出层
'''

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: reuse_ pretrained_layers.py
@time: 2018/8/27 14:15
"""

import tensorflow as tf

n_hidden2 = 200 # new layer
n_output = 10 # new output


# 图的结构信息 保存在.meta文件信息中
saver = tf.train.import_meta_graph(r'../model/my_model_final.ckpt.meta')
for op in tf.get_default_graph().get_operations():
    print(op.name)

# 根据 op的名字<op_name : op_index> 来获取原来的tensor
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

# 获取hidden1
hidden1 = tf.get_default_graph().get_tensor_by_name("")

# reuse tf 模型
# with tf.Session() as sess:
#     saver.restore(sess,r'../model/my_model_final.ckpt')




