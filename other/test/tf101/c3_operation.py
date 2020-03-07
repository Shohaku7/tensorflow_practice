import tensorflow as tf
# 常量操作
a = tf.constant(2)
b = tf.constant(3)
# 创建会话，并执行计算操作
with tf.Session() as sess:
    print("a: %i" % sess.run(a))
    print("b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))
# 占位符操作
# x = tf.placeholder(dtype, shape, name)
x = tf.placeholder(tf.int16, shape=(), name="x")
y = tf.placeholder(tf.int16, shape=(), name="y")
# 计算操作
add = tf.add(x, y)
mul = tf.multiply(x, y)
# 加载默认数据流图
with tf.Session() as sess:
    # 不填充数据，直接执行操作，报错
    print("Addition with variables: %i" % sess.run(add, feed_dict={x: 10, y: 5}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={x: 2, y: 3}))
# 加载默认数据流图
with tf.Session() as sess:
    # 不填充数据，直接执行操作，报错
    print("Addition with variables: %i" % sess.run(add, feed_dict={x: 2, y: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={x: 2, y: 3}))
