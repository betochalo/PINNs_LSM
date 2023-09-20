# import tensorflow as tf
import torch

# x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# x = tf.ones((3, 3))
# x = tf.zeros((2, 3))
# x = tf.eye(3)
# x = tf.random.normal((3, 3), mean=0, stddev=1)
# x = tf.random.uniform((1, 3), minval=0, maxval=1)
# x = tf.range(start=1, limit=10, delta=2)
# x = tf.cast(x, dtype=tf.float64)
# x = tf.constant([1, 2, 3])
# y = tf.constant([4, 5, 6])
# z = tf.add(x, y)
# z = x * y
# z = tf.tensordot(x, y, axes=1)
# a = tf.random.normal((2, 3))
# b = tf.random.normal((3, 4))
# c = tf.matmul(a, b)  # a @ b
#
# # indexing
#
# g = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
# # print(g[:])
# # print(g[1:])
# # print(g[1:3])
# # print(g[::2])
# # print(g[::-1])
#
# indices = tf.constant([0, 3])
# x_ind = tf.gather(g, indices)
#
# h = tf.constant([[1, 2],
#                 [3, 4],
#                 [5, 6]])
# # print(h[0, :])
# # print(h[0:2, :])
#
# # reshape
# x = tf.range(9)
# x = tf.reshape(x, (3, 3))
# x = tf.transpose(x, perm=[1, 0])
# print(x)
c = torch.concat([torch.ones(200, 1), torch.zeros(200, 1)], dim=1)
print(c)


