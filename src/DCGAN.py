import tensorflow as tf
import numpy as np




def residualBlock(x, numFilters, kernelSize, isTraining):
	print()
	shape = x.get_shape().as_list()
	if shape[-1] == numFilters:
		shortcut = x
	else:
		output = x
		output = tf.transpose(output, [0,3,1,2])
		print(output)

		output = tf.concat([output, output, output, output], axis=1)
		output = tf.transpose(output, [0,2,3,1])
		output = tf.depth_to_space(output, 2)
		output = tf.transpose(output, [0,3,1,2])
		output = tf.layers.conv2d(output, numFilters, 1, 1, padding='SAME', use_bias=True)
		print(output)
		exit()
	print(shortcut)


	output = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=isTraining)
	print(output)
	output = tf.nn.relu(output)
	print(output)

	output = tf.layers.conv2d(output, numFilters, kernelSize, 1, padding='SAME', use_bias=False, kernel_initializer='he_normal')
	print(output)
	output = tf.layers.batch_normalization(output, momentum=0.9, epsilon=1e-5, training=isTraining)
	print(output)
	output = tf.nn.relu(output)
	print(output)
	output = tf.layers.conv2d(output, numFilters, kernelSize, 1, padding='SAME', use_bias=True, kernel_initializer='he_normal')
	print(output)

	return shortcut + output


def generator(x, isTraining, dim=64, kernelSize=5):
    
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		print(x)

		x = tf.layers.dense(x, 4*4*8*dim)
		print(x)
		
		x = tf.reshape(x, [-1, 4, 4, 8*dim]) 
		print(x)

		#res1 = residualBlock(reshaped, 8*dim, 3, isTraining)
		#res2 = residualBlock(reshaped, 4*dim, 3, isTraining)

		x = tf.nn.relu(tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=isTraining))
		print(x)

		x = tf.layers.conv2d_transpose(x, 4*dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=isTraining))
		print(x)

		x = tf.layers.conv2d_transpose(x, 2*dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=isTraining))
		print(x)
		
		x = tf.layers.conv2d_transpose(x, dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=isTraining))
		print(x)

		x = tf.tanh(tf.layers.conv2d_transpose(x, 3, kernelSize, 2, padding='SAME', kernel_initializer='he_normal'))
		print(str(x) + "\n\n")

		return x


def discriminator(x, isTraining, dim=64, kernelSize=5):
    
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		print(x)

		x = tf.layers.conv2d(x, dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(x)
		print(x)

		x = tf.layers.conv2d(x, 2*dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(x)
		print(x)

		x = tf.layers.conv2d(x, 4*dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(x)
		print(x)

		x = tf.layers.conv2d(x, 8*dim, kernelSize, 2, padding='SAME', use_bias=True, kernel_initializer='he_normal')
		x = tf.nn.relu(x)
		print(x)

		x = tf.layers.flatten(x)
		print(x)

		x = tf.layers.dense(x, 1, kernel_initializer='he_normal')
		print(str(x) + "\n\n")

		return x