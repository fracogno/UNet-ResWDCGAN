import tensorflow as tf
import numpy as np


def generator(Z, isTraining):
    
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		print(Z)

		size, size3 = 4, 1024
		x = tf.reshape(tf.layers.dense(Z, size*size*size3), [-1, size, size, size3])
		x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTraining))
		print(x)

		filters = [512, 256, 128, 64]
		for numFilters in filters:
			x = tf.layers.conv2d_transpose(x, numFilters, 3, 2, padding='SAME')
			x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTraining))
			print(x)

		output = tf.tanh(tf.layers.conv2d_transpose(x, 3, 3, 1, padding='SAME'))
		print(output)
		print("\n\n")

		return output


def discriminator(x, isTraining):
    
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		print(x)
	
		filters = [64, 128, 256, 512]
		for numFilters in filters:
			x = tf.layers.conv2d(x, numFilters, 3, 2, padding='SAME')

			if numFilters != filters[0]:
				x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTraining))
			else:
				x = tf.nn.leaky_relu(x)
			print(x)

		x = tf.nn.leaky_relu(tf.layers.dense(tf.layers.flatten(x), 128))
		print(x)

		x = tf.layers.dense(x, 1)
		print(str(x) + "\n")

		return x