import tensorflow as tf


def downsample(inputs, filters, size, isTraining, apply_batchnorm=True):
    
    result = tf.layers.conv2d(inputs, filters, size, strides=2, padding='SAME', use_bias=False, 
                              kernel_initializer=tf.random_normal_initializer(0., 0.02))
    
    if apply_batchnorm:
        result = tf.layers.batch_normalization(result, training=isTraining)

    return tf.nn.leaky_relu(result)


def upsample(inputs, filters, size, isTraining, apply_dropout=False):
    
    result = tf.layers.conv2d_transpose(inputs, filters, size, strides=2, padding='SAME', use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))
    
    result = tf.layers.batch_normalization(result, training=isTraining)

    if apply_dropout:
        result = tf.nn.dropout(result, keep_prob=0.5)

    return tf.nn.relu(result)
    
    
def getAutoencoder(X, isTraining):
    filters = [32, 64, 128, 256, 512, 512]
    
    with tf.variable_scope('autoencoder', reuse=tf.AUTO_REUSE):
        output = X
        print(output)
        skips = []
        # Encoder
        for num_f in filters:
            output = downsample(output, num_f, 5, isTraining, num_f != filters[0])
            skips.append(output)
            print(output)
        
        # Decoder
        skips = reversed(skips[:-1])
        for num_f, skip in zip(reversed(filters[:-1]), skips):
            output = upsample(output, num_f, 5, isTraining, apply_dropout=num_f == 512)
            output = tf.concat([output, skip], axis=3)
            print(output)
        
        # Additional layers (output is bigger than input in this autoencoder)
        output = upsample(output, 64, 5, isTraining, apply_dropout=False)
        print(output)
        output = upsample(output, 32, 5, isTraining, apply_dropout=False)
        print(output)
            
        # Last layer => Only 3 channels and TANH
        last = tf.layers.conv2d_transpose(output, 3, 5, strides=2, padding='SAME', activation="tanh",
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02))
        print(last)    
        print()

        down1 = tf.image.resize_images(last, (64, 64))
        print(down1)

        print()
        return last, down1


def getAutoencoderDiscriminator(X, isTraining):

    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
        initializer = tf.random_normal_initializer(0., 0.02)

        down1 = downsample(X, 64, 4, isTraining, False)
        down2 = downsample(down1, 128, 4, isTraining)
        down3 = downsample(down2, 256, 4, isTraining)
        print(X)
        print(down1)
        print(down2)
        print(down3)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
        conv = tf.layers.conv2d(zero_pad1, 512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        bn = tf.layers.batch_normalization(conv, training=True)
        lrelu = tf.nn.leaky_relu(bn)
        print(lrelu)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(lrelu) 
        last = tf.layers.conv2d(zero_pad2, 1, 4, strides=1, kernel_initializer=initializer)
        print(last)
        print()
        return last
