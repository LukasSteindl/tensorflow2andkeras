#import tensorflow as tf 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
message = tf.constant('Welcome')

in_a = tf.placeholder(dtype=tf.float32,shape=(2))

def model(x):
    with tf.variable_scope("matmul"):
        W = tf.get_variable("W",initializer=tf.ones(shape=(2,2)))
        b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
        return x*W + b
    
out_a = model(in_a)

with tf.Session() as sess:
    print(sess.run(message).decode())
    writer = tf.summary.FileWriter("./logs/example",sess.graph)


