import tensorflow as tf


def WGAN_Loss(D_logits_real, D_logits_fake, grad_penalty):
    # Maximize REAL - FAKE === minimize FAKE - REAL
    D_loss = tf.reduce_mean(D_logits_fake) - tf.reduce_mean(D_logits_real) + grad_penalty
    G_loss = - tf.reduce_mean(D_logits_fake)

    return D_loss, G_loss


def WGAN_Optimizer(D_loss, G_loss, lr, beta1, beta2):
    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    with tf.control_dependencies(D_update_ops):
        D_optimizer = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))    

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(G_update_ops):
        G_optimizer = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

    return D_optimizer, G_optimizer


def GAN_Loss(D_logits_real, D_logits_fake, label_smoothing):
    # Discriminator
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real,
                                                labels=tf.ones_like(D_logits_real) * label_smoothing))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.zeros_like(D_logits_fake)))
    D_loss = D_loss_real + D_loss_fake

    # Generator
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.ones_like(D_logits_fake)))

    return D_loss, G_loss


def Autoencoder_Optimizer(C_loss, AE_loss, lr, beta1):
    C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='critic')
    with tf.control_dependencies(C_update_ops):
        C_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(C_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))    

    # Optimize both small generator and UNET
    '''AE_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='autoencoder')
    with tf.control_dependencies(AE_update_ops):
    	AE_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(AE_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder'))'''
    
    AE_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='autoencoder')
    with tf.control_dependencies(AE_update_ops):
        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        with tf.control_dependencies(G_update_ops):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            AE_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(AE_loss, var_list=train_vars)
            
    return C_optimizer, AE_optimizer
