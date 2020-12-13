import tensorflow as tf

def linear_layer(x):
    return 3*x+2

@tf.function
def simple_nn(x):
    return tf.nn.relu(linear_layer(x))

def simple_function(x):
    return 3*x

print(simple_nn)