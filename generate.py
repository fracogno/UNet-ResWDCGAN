import tensorflow as tf
import numpy as np
import src.network as network, src.UNET_GAN as UNET_GAN, src.util as util
import matplotlib.pyplot as plt


basePath = "/home/francesco/UQ/TMP/StackWDCGAN/"
Z_dim = 128

Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

G_z = network.generator(Z, isTraining)
G_AE, _ = UNET_GAN.getAutoencoder(G_z, isTraining)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, basePath + "checkpoints/ckpt-G-")
    saver.restore(sess, basePath + "checkpoints/ckpt-AE-80250")

    images = []
    for j in range(100):
        G_output = sess.run(G_AE, feed_dict={ isTraining : False, Z : network.sample_noise([1, Z_dim]) })
        images.append(G_output[0])
    images = np.array(images)
    print(images.shape)
    util.saveImages(basePath + "images/TEST-" + str(0), images)
