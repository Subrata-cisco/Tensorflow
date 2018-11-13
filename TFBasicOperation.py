import tensorflow as tf

matFill  = tf.fill((4,4),10)
zero = tf.zeros((4,4))

with tf.Session() as sess:
    pp = sess.run(zero)
    print(pp)

allOnce = tf.ones((4,4))

normal = tf.random_normal((4,4),mean=0,stddev=1.0)
uniform = tf.random_uniform((4,4), minval=0,maxval=5)

with tf.Session() as sess:
    gn = sess.run(normal)
    print(gn)
    print("\n")
    gu = sess.run(uniform)
    print(gu)
    print("\n")

# or the following way..

sess = tf.InteractiveSession()
myList = [matFill,zero,allOnce,normal,uniform]

for op in myList:
    # print(sess.run(op))
    print(op.eval())
    print("\n")
