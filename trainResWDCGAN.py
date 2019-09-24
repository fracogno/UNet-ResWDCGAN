import tensorflow as tf
import src.util as util, src.DCGAN as ResWDCGAN, src.losses as losses

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
basePath = "/scratch/cai/UNet-ResWDCGAN/"
#basePath = "/home/francesco/UQ/UNet-ResWDCGAN/"

'''import pickle
with open(basePath + "dataset/TMP.pkl", 'rb') as handle:
	X = pickle.load(handle)'''

# Get data
imgSize = 64
X = util.getData(basePath + "dataset/NvAndMelNoDuplicatesFullSize.zip", imgSize, value="nv")
assert(X.max() == 1. and X.min() == -1.)
print(X.shape)

# Parameters
epochs = 30000
batchSize = 64
lr = 1e-4
lam = 10.0
num_D = 5
beta1 = 0.5
beta2 = 0.9
Z_dim = 128

# Dataset
def generator():
	for el in X:
		yield el

dataset = tf.data.Dataset.from_generator(generator, (tf.float32),
					output_shapes=(tf.TensorShape([imgSize, imgSize, 3]))).repeat(epochs).shuffle(buffer_size=len(X)).batch(batchSize, drop_remainder=True)
iterator = dataset.make_initializable_iterator()
X_tensor = iterator.get_next()

Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

''' 
	Networks
'''
G_z = ResWDCGAN.generator(Z, isTraining)
D_logits_real = ResWDCGAN.discriminator(X_tensor, isTraining)
D_logits_fake = ResWDCGAN.discriminator(G_z, isTraining)

# Compute gradient penalty
eps = tf.random_uniform([batchSize, 1, 1, 1], minval=0., maxval=1.)
X_inter = eps * X_tensor + (1. - eps) * G_z
grad = tf.gradients(ResWDCGAN.discriminator(X_inter, isTraining), [X_inter])[0]
gradients = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
grad_penalty = lam * tf.reduce_mean(tf.square(gradients - 1.))

# Losses and optimizers
D_loss, G_loss, = losses.WGAN_Loss(D_logits_real, D_logits_fake, grad_penalty)
D_optimizer, G_optimizer = losses.WGAN_Optimizer(D_loss, G_loss, lr, beta1, beta2)

# Tensorboard | VISUALIZE => tensorboard --logdir=.
summaries_dir = basePath + "checkpoints"
merged_summary = tf.summary.merge([tf.summary.scalar('D_loss', -D_loss), tf.summary.scalar('G_loss', -G_loss)])

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	saver = tf.train.Saver(max_to_keep=10000)
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())

	GStep = 0
	sess.run(iterator.initializer)
	try:
		while True:
			# Sample Gaussian noise
			noise = util.sample_noise([batchSize, Z_dim])
			
			# Train discriminator (more at the beginning)
			D_iterations = 30 if (GStep < 5 or GStep % 500 == 0) else num_D
			for _ in range(D_iterations):
				_, summary = sess.run([D_optimizer, merged_summary], feed_dict={ isTraining: True, Z: noise })
				summary_writer.add_summary(summary, GStep)					

			# Train Generator
			sess.run(G_optimizer, feed_dict={ isTraining: True, Z: noise })

			# Save checkpoint and generated images at this step
			if GStep % 1000 == 0:
				saver.save(sess, basePath + "checkpoints/ckpt-ResWDCGAN-" + str(GStep))
				output = sess.run(G_z, feed_dict={ isTraining : False, Z: util.sample_noise([10, Z_dim]) })
				util.saveImages(basePath + "images/out-ResWDCGAN-" + str(GStep), output)
			GStep += 1
	except tf.errors.OutOfRangeError:
		pass
