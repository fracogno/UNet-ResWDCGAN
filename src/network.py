import tensorflow as tf
import numpy as np


def generator(Z, isTraining, kernelSize=5):
    
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		print(Z)

		size, size3 = 4, 1024
		x = tf.reshape(tf.layers.dense(Z, size*size*size3), [-1, size, size, size3])
		x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTraining))
		print(x)

		x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(x, 512, kernelSize, 1, padding='SAME', use_bias=False), training=isTraining))
		print(x)

		x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(x, 512, kernelSize, 2, padding='SAME', use_bias=False), training=isTraining))
		print(x)

		x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(x, 256, kernelSize, 2, padding='SAME', use_bias=False), training=isTraining))
		print(x)

		x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(x, 128, kernelSize, 2, padding='SAME', use_bias=False), training=isTraining))
		print(x)

		x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(x, 64, kernelSize, 2, padding='SAME', use_bias=False), training=isTraining))
		print(x)

		x = tf.tanh(tf.layers.conv2d_transpose(x, 3, kernelSize, 1, padding='SAME'))
		print(x)
		print("\n\n")

		return x


def discriminator(x, isTraining, kernelSize=5):
    
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		print(x)

		x = tf.nn.leaky_relu(tf.layers.conv2d(x, 64, kernelSize, 2, padding='SAME'))
		print(x)

		x = tf.nn.leaky_relu(tf.layers.conv2d(x, 128, kernelSize, 2, padding='SAME'))
		print(x)

		x = tf.nn.leaky_relu(tf.layers.conv2d(x, 256, kernelSize, 2, padding='SAME'))
		print(x)

		x = tf.nn.leaky_relu(tf.layers.conv2d(x, 512, kernelSize, 2, padding='SAME'))
		print(x)

		x = tf.layers.flatten(x)
		print(x)

		x = tf.layers.dense(x, 1)
		print(str(x) + "\n")

		return x