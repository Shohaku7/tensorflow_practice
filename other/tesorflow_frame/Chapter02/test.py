# 　　import加载的模块分为四个通用类别：
#
# 　　a.使用python编写的代码（.py文件）；
#
# 　　b.已被编译为共享库或DLL的C或C++扩展；
#
# 　　c.包好一组模块的包
#
# 　　d.使用C编写并链接到python解释器的内置模块；
import tensorflow as tf
import tensorflow.contrib as aa
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a + b
sess = tf.Session()
sess.run(result)
bb = tf.contrib
cc = aa.ffmpeg
dd = bb.ffmpeg

print(tf)
print(aa)
print(type(aa))
print(bb)
print(type(bb))

# print(cc)
# print(dd)
