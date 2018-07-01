'''
  使用TensorFlow的高级API 训练MLP
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 将数字图片转化成机器能识别的数字 格式
digits = datasets.load_digits() # 加载sklearn 自带的手写数字图片数据集
print(digits.images[0])
print(digits.data[0])
print(digits.keys())
# digits.images : 共有1797张 表示为 8*8像素的图片 数字矩阵的表示
# digits.data : 1797张数字图片的一维矩阵的表现形式
# digits.target : [0,1,2,3,4,5,6,7,8,9] 数字标签 也就是每张图片表示的数字

# 可视化 数字
def display_img(img_no):
	fig,ax = plt.subplots(figsize=(6, 6))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.imshow(digits.images[img_no],cmap = plt.cm.binary)
	ax.matshow(digits.images[img_no],cmap = plt.cm.binary) #Display an array as a matrix in a new figure window.
	plt.show()
# display_img(3)
X = digits.data
y = digits.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.3)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train_scaled)
dnn_classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_columns)
dnn_classifier.fit(x=X_train_scaled,y=y_train,batch_size=50,steps=40000)

from sklearn.metrics import accuracy_score
y_pred = dnn_classifier.predict(scaler.transform(X_test))
accuracy_score = accuracy_score(y_test, list(y_pred))
print('DNN准确率：',accuracy_score)




















