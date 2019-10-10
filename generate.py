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
    saver.restore(sess, basePath + "nv-checkpoints/ckpt-AE-79500")
    #saver.restore(sess, basePath + "mel-checkpoints/ckpt-AE-7500")

    images, fullImg = [], []
    for j in range(30):
        UNET_output, G_output = sess.run([G_AE, G_z], feed_dict={ isTraining : False, Z : util.sample_noise([1, Z_dim]) })
        images.append(G_output[0])
        fullImg.append(UNET_output[0])
    images = np.array(images)
    fullImg = np.array(fullImg)
    print(images.shape)
    print(fullImg.shape)
    util.saveImages(basePath + "images/TEST-" + str(0), images)
    util.saveImages(basePath + "images/TEST-AE-" + str(0), fullImg)
