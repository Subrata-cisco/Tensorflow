import tensorflow as tf

mat = tf.random_uniform((4,4),0,1)
variable = tf.Variable(mat)

sess = tf.InteractiveSession();
# sess.run(variable)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(variable))

ph = tf.placeholder(dtype=tf.float32)