import tensorflow as tf
import src.util as util, src.DCGAN as ResWDCGAN, src.UNetGAN as UNetGAN, src.losses as loss

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
basePath = "/scratch/cai/UNet-ResWDCGAN/"
#basePath = "/home/francesco/UQ/UNet-ResWDCGAN/"

'''import pickle
with open(basePath + "dataset/TMP.pkl", 'rb') as handle:
	X = pickle.load(handle)'''

# Get data in different sizes
imgSize = 256
X = util.getData(basePath + "dataset/NvAndMelNoDuplicatesFullSize.zip", imgSize, value="nv")
print(X.shape)
assert(X.max() == 1. and X.min() == -1.)

# Parameters
epochsAE = 30000
batchSize = 64
lr = 1e-4
Z_dim = 128
l1_weight = 100.

# Dataset
def generator():
	for el in X:
		yield el

dataset = tf.data.Dataset.from_generator(generator, (tf.float32),
					output_shapes=(tf.TensorShape([imgSize, imgSize, 3]))).repeat(epochsAE).shuffle(buffer_size=len(X)).batch(batchSize, drop_remainder=True)
iterator = dataset.make_initializable_iterator()
X_tensor = iterator.get_next()

Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

''' 
	Networks
'''
G_z = ResWDCGAN.generator(Z, isTraining)
G_AE, downsized = UNetGAN.getAutoencoder(G_z, isTraining)
C_logits_real = UNetGAN.getAutoencoderDiscriminator(X_tensor, isTraining)
C_logits_fake = UNetGAN.getAutoencoderDiscriminator(G_AE, isTraining)

# Losses and optimizer
C_loss, AE_gan = loss.GAN_Loss(C_logits_real, C_logits_fake, 0.9)
AE_L1 = l1_weight * tf.reduce_mean(tf.abs(G_z - downsized)) 
AE_loss = AE_gan + AE_L1

C_optimizer, AE_optimizer = loss.Autoencoder_Optimizer(C_loss, AE_loss, lr, 0.5)

# Tensorboard | VISUALIZE => tensorboard --logdir=.
summaries_dir = basePath + "checkpoints"
AE_summaries = [tf.summary.scalar('C_loss', C_loss), tf.summary.scalar('AE_L1', AE_L1), tf.summary.scalar('AE_gan', AE_gan)]
AE_merged_summary = tf.summary.merge(AE_summaries)

# Training
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	saver = tf.train.Saver(max_to_keep=3000)
	sess.run(tf.global_variables_initializer())

	# Load pretrained generator 64x64
	saverG = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
	saverG.restore(sess, basePath + "checkpoints/ckpt-ResWDCGAN-183000")

	AE_summary_writer = tf.summary.FileWriter(summaries_dir,  graph=tf.get_default_graph())

	# Train AE GAN
	AEStep = 0
	sess.run(iterator.initializer)
	try:
		while True:
			noise = util.sample_noise([batchSize, Z_dim])

			# Train Critic
			_, summary = sess.run([C_optimizer, AE_merged_summary], feed_dict={ isTraining: True, Z: noise })
			AE_summary_writer.add_summary(summary, AEStep)
			
			# Train AE only
			sess.run(AE_optimizer, feed_dict={ isTraining: True, Z: noise })

			if AEStep % 750 == 0:
				saver.save(sess, basePath + "checkpoints/ckpt-AE-" + str(AEStep))
				AE_output = sess.run(G_AE, feed_dict={ isTraining : False, Z: util.sample_noise([4, Z_dim]) })
				util.saveImages(basePath + "images/out-AE-" + str(AEStep), AE_output)
			AEStep += 1
	except tf.errors.OutOfRangeError:
		pass
