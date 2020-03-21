import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
#
# # 导入数据
# mnist = input_data.read_data_sets('./mnist/dataset/')

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist/mnist.npz')