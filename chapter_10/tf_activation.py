''' tf 中的激活函数的实现'''

"""
@version: 1.0.0
@author: zhengjw
@Mail: zhengjw@2345.com
@file: tf_activation.py
@time: 2018/8/27 11:51
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# relu
def relu(z):
	return np.maximum(0,z)

# leak relu
def leak_relu(z,alpha=0.01):
	return np.maximum(alpha * z ,z)

# elu

def elu(z,alpha = 1):
	return np.where( z < 0,alpha * (np.exp(z) - 1),z)

# logit
def logit(z):
	return 1/(1 +np.exp(-z))


if __name__ == '__main__':
	# logit
	z = np.linspace(-5,5.200) #生成 -5 ，5 的200个等差点
	plt.plot([-5, 5], [0, 0], 'k-')
	plt.plot([-5, 5], [1, 1], 'k--')
	plt.plot([0, 0], [-0.2, 1.2], 'k-')
	plt.plot([-5, 5], [-3 / 4, 7 / 4], 'g--')
	plt.plot(z, logit(z), "b-", linewidth=2)
	props = dict(facecolor='black', shrink=0.1)
	plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
	plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
	plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")
	plt.grid(True)
	plt.title("Sigmoid activation function", fontsize=14)
	plt.axis([-5, 5, -0.2, 1.2])

	# save_fig("sigmoid_saturation_plot")
	plt.show()




