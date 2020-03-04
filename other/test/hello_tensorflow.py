import tensorflow as tf
# 定义常量操作 hello
hello = tf.constant("hello")
# 创建一个会话
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# 执行常量操作 hello 并打印到标准输出
print(sess.run(hello))