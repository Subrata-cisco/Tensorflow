import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

#print(x_data)
#print(y_data)
#print(np.random.rand(2))

# y = mx + c
m = tf.Variable(.16)
c = tf.Variable(.45)

error = 0
for x,y in zip(x_data,y_data):
    y_hat = m*x + c
    error += (y - y_hat)**2


optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess :
    sess.run(init)

    training_steps = 100
    for i in range(training_steps):
        sess.run(train)

    final_scope,final_intercept = sess.run([m,c])

x_test = np.linspace(-1,11,10)
y_pred_plot = final_scope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,"r")
plt.plot(x_data,y_data,"*")
plt.show()

