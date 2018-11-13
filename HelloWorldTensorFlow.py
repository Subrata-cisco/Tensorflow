import tensorflow as tf

print("Hello World from Tensorflow ::",tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("World")

print(type(hello))

with tf.Session() as sess:
    result = sess.run(hello+world)
    print(result)

