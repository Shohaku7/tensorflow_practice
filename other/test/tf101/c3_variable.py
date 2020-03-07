import tensorflow as tf
# 创建变量
# tf.random_normal 方法返回形状为(1，4)的张量。它的4个元素符合均值为100、标准差为0.35的正态分布。
# random_normal正态分布
W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name="W")
b = tf.Variable(tf.zeros([4]), name="b")
[W, b]
# W b 权重和偏置
# 初始化变量
# 创建会话（之后小节介绍） 通过会话获取资源
sess = tf.Session()
# 使用 global_variables_initializer 方法初始化全局变量 W 和 b
sess.run(tf.global_variables_initializer())
# 执行操作，获取变量值
sess.run([W, b])
# 执行更新变量 b 的操作
sess.run(tf.assign_add(b, [1, 1, 1, 1]))
# 查看变量 b 是否更新成功
sess.run(b)
# 创建Saver
saver = tf.train.Saver({'W': W, 'b': b})
# 存储变量到文件 './summary/test.ckpt-0'
saver.save(sess, './summary/test.ckpt', global_step=0)
# 再次执行更新变量 b 的操作
sess.run(tf.assign_add(b, [1, 1, 1, 1]))
# 获取变量 b 的最新值
sess.run(b)
# 从文件中恢复变量 b 的值
saver.restore(sess, './summary/test.ckpt-0')
# 查看变量 b 是否恢复成功
sess.run(b)
# 从文件中恢复数据流图结构
# tf.train.import_meta_graph