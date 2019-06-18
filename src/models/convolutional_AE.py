import tensorflow as tf

class Encoder(tf.keras.models.Model):
    '''
    Simple autoencoder forward layer
    '''
    
    def __init__(self, latent_size = 10):
        tf.keras.models.Model.__init__(self)
        self._latent_size = latent_size

        # Apply a 5x5 kernel to the image:
        self.encoder_layer_1 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5], 
            filters     = 24,
            activation  = tf.nn.relu,
        )
        
        # Use a 2x2 kernel of stride 2x2 to downsample:
        self.encoder_layer_2 = tf.keras.layers.Convolution2D(
            kernel_size = [3, 3],
            strides     = [3, 3],
            filters     = 48,
            activation=tf.nn.relu,
        )
        
        self.encoder_layer_3 = tf.keras.layers.Convolution2D(
            kernel_size = [5, 5], 
            filters     = 64, 
            activation  = tf.nn.relu,
        )
        
        # The final shape at this point is 8 x (8*num_digits + 6*(num_digits - 1) )

        self.encoder_layer_4 = tf.keras.layers.Convolution2D(
            kernel_size = [3, 3],
            strides     = [3, 3],
            filters     = latent_size,
            activation  = None,
        )

        # The final shape at this point is 4 x (4*num_digits + 3*(num_digits - 1) )

        # Use a global average pooling layer to get to 10 numbers per image:
        

    def call(self, inputs):
        
        batch_size = inputs.shape[0]

        x = self.encoder_layer_1(inputs)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_layer_4(x)
        return tf.reshape(x, [batch_size, self._latent_size])

        

class Decoder(tf.keras.models.Model):
    
    def __init__(self):
        tf.keras.models.Model.__init__(self)


        # The decoder runs the encoder steps but in reverse.

        self.decoder_layer_1 = tf.keras.layers.Convolution2DTranspose(
            kernel_size    = [3, 3],
            strides        = [3, 3], 
            output_padding = [1,1],
            filters        = 64,
            activation     = tf.nn.relu
        )

        self.decoder_layer_2 = tf.keras.layers.Convolution2DTranspose(
            kernel_size = [5, 5],
            filters     = 48,
            activation  = tf.nn.relu
        )

        self.decoder_layer_3 = tf.keras.layers.Convolution2DTranspose(
            kernel_size = [3, 3],
            strides     = [3, 3], 
            filters     = 24,
            activation  = tf.nn.relu
        )

        self.decoder_layer_4 = tf.keras.layers.Convolution2DTranspose(
            kernel_size = [5, 5],
            filters     = 1,
        )




        
    def call(self, inputs):

            
        batch_size = inputs.shape[0]

        # First Step is to to un-pool the encoded state into the right shape:
        x = tf.reshape(inputs, [batch_size, 1, 1, inputs.shape[-1]])

        x = self.decoder_layer_1(x)
        x = self.decoder_layer_2(x)
        x = self.decoder_layer_3(x)
        x = self.decoder_layer_4(x)
        return x
        