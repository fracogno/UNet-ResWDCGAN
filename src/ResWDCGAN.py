import tensorflow as tf
import numpy as np


def sample_noise(size, mu=0., sigma=1.):
    return np.random.normal(mu, sigma, size=size)


def upsample(inputs):
  _, nh, nw, _ = inputs.get_shape().as_list()
  return tf.image.resize_nearest_neighbor(inputs, [nh * 2, nw * 2])


def downsample(inputs):
  return tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')


def residualBlockGenerator(inputs, filters, isTraining):
	# First part
	relu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(inputs, training=isTraining))
	print(relu1)

	upsampled = upsample(relu1)
	print(upsampled)

	conv1 = tf.layers.conv2d(upsampled, filters, 3, 1, 'SAME')
	print(conv1)

	relu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTraining))
	print(relu2)

	conv2 = tf.layers.conv2d(relu2, filters, 3, 1, 'SAME')
	print(conv2)

	# Second part
	upsampledInput = upsample(inputs)
	print(upsampledInput)

	convInput = tf.layers.conv2d(upsampledInput, filters, 1, 1, 'SAME')
	print(convInput)

	# Skip connection
	output = conv2 + convInput
	print(output)
	print()

	return output


def residualBlockDiscriminator(inputs, filters, isTraining):
	conv1 = tf.nn.leaky_relu(tf.layers.conv2d(inputs, filters, 3, 1, 'SAME'))
	print(conv1)

	conv2 = tf.layers.conv2d(conv1, filters, 3, 1, 'SAME')
	print(conv2)

	downsampled = downsample(conv2)
	print(downsampled)

	# Second part
	downsampledInput = downsample(inputs)
	print(downsampledInput)

	convInput = tf.layers.conv2d(downsampledInput, filters, 3, 1, 'SAME')
	print(convInput)

	# Skip connection
	output = downsampled + convInput
	print(output)

	output = tf.nn.leaky_relu(output)
	print(output)
	print()

	return output


def generator(Z, isTraining):
    
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		print(Z)

		size, size3 = 4, 1024
		output = tf.reshape(tf.layers.dense(Z, size*size*size3), [-1, size, size, size3])
		print(output)
		
		filters = [512, 256, 128, 64, 32]
		for numFilters in filters:
			output = residualBlockGenerator(output, numFilters, isTraining)

		relu = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=isTraining))
		print(relu)

		output = tf.tanh(tf.layers.conv2d_transpose(relu, 3, 3, 1, padding='SAME'))
		print(output)
		print("\n\n")

		return output


def discriminator(X, isTraining):
    
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		print(X)
		output = X
	
		filters = [32, 64, 128, 256, 512]
		for numFilters in filters:
			output = residualBlockDiscriminator(output, numFilters, isTraining)
			print(output)

		output = (tf.layers.dense(tf.layers.flatten(output), 128))
		print(output)

		output = tf.layers.dense(output, 1)
		print(output)
		print("\n\n")

		return output
