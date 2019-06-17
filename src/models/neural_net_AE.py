import tensorflow as tf

class Encoder(tf.keras.models.Model):
    '''
    Simple autoencoder forward layer
    '''
    
    def __init__(self, latent_size=10):
        tf.keras.models.Model.__init__(self)
        self._latent_size = 10
        
        self.encoder_layer_1 = tf.keras.layers.Dense(
            units = 784, 
            activation=tf.nn.relu,
        )
        self.encoder_layer_2 = tf.keras.layers.Dense(
            units = 392,
            activation = tf.nn.relu,
        )
        self.encoder_layer_3 = tf.keras.layers.Dense(
            units = 98,
            activation = tf.nn.relu,
        )
        
        self.final_encoding_layer = tf.keras.layers.Dense(
            units = self._latent_size,
            activation = None,
        )
        
    def call(self, inputs):
        
        x = self.encoder_layer_1(inputs)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        return self.final_encoding_layer(x)
        
        

class Decoder(tf.keras.models.Model):
    
    def __init__(self, num_digits):
        tf.keras.models.Model.__init__(self)


        self.decoder_layer_1 = tf.keras.layers.Dense(
            units = 98, 
            activation=tf.nn.relu,
        )
        self.decoder_layer_2 = tf.keras.layers.Dense(
            units = 392,
            activation = tf.nn.relu,
        )
        self.decoder_layer_3 = tf.keras.layers.Dense(
            units = 784,
            activation = tf.nn.relu,
        )
        
        self.final_decoding_layer = tf.keras.layers.Dense(
            units = num_digits*784,
        )
        
    def call(self, inputs):
        
        x = self.decoder_layer_1(inputs)
        x = self.decoder_layer_2(x)
        x = self.decoder_layer_3(x)
        return self.final_decoding_layer(x)
        