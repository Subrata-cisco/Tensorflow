import tensorflow as tf

graph_one = tf.get_default_graph()

graph_two = tf.Graph()

print(graph_one)
print(graph_two)

print(graph_one is tf.get_default_graph)
print(graph_two is tf.get_default_graph)

with graph_two.as_default():
    print(graph_two is tf.get_default_graph())