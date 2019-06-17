import tensorflow as tf

class mnist_classifier(tf.keras.models.Model):
    '''
    Simple autoencoder forward layer
    '''
    
    def __init__(self, n_output):
        tf.keras.models.Model.__init__(self)

        # Apply a 5x5 kernel to the image:
        self.layer_1 = tf.keras.layers.Convolution2D(
            kernel_size = [3, 3], 
            filters     = 32,
            activation  = tf.nn.relu,
        )
        
        # Use a 2x2 kernel of stride 2x2 to downsample:
        self.layer_2 = tf.keras.layers.Convolution2D(
            kernel_size = [3, 3],
            filters     = 64,
            activation=tf.nn.relu,
        )
        
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.layer_3 = tf.keras.layers.Convolution2D(
            kernel_size = [3,3],
            filters = 128,
            activation = tf.nn.relu,
        )

        self.layer_4 = tf.keras.layers.Convolution2D(
            kernel_size = [3,3],
            filters = 10,
            activation = None,
        )

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        # The final shape at this point is 4 x (4*num_digits + 3*(num_digits - 1) )

        # Use a global average pooling layer to get to 10 numbers per image:
        

    def call(self, inputs):
        
        x = inputs
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.pool(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.pool2(x)
        return x