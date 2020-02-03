import tensorflow as tf
import numpy as np
import src.DCGAN as network, src.UNetGAN as UNET_GAN, src.util as util
import matplotlib.pyplot as plt


basePath = "/scratch/cai/UNet-ResWDCGAN/"
Z_dim = 128

Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

G_z = network.generator(Z, isTraining)
G_AE, _ = UNET_GAN.getAutoencoder(G_z, isTraining)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, basePath + "checkpoints/ckpt-AE-101250")
    #saver.restore(sess, basePath + "checkpoints-mel/ckpt-AE-88500")

    images, fullImg = [], []
    for j in range(50):
        UNET_output, G_output = sess.run([G_AE, G_z], feed_dict={ isTraining : False, Z : util.sample_noise([1, Z_dim]) })
        images.append(G_output[0])
        fullImg.append(UNET_output[0])

    images = np.array(images)
    fullImg = np.array(fullImg)
    print(images.shape)
    print(fullImg.shape)

    util.saveImages(basePath + "generated/TEST-" + str(0), images)
    util.saveImages(basePath + "generated/TEST-AE-" + str(0), fullImg)