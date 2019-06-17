import tensorflow as tf
import random
import numpy


class mnist_generator(object):
    '''
    This class takes the mnist dataset and generates multi-digit examples.
    The goal here is to create on-the-fly augmented data that is more complex
    than just 0 to 9, but also very easy to get access to.
    '''
    
    def __init__(self, seed=0):
        # Use TF to get the dataset, will download if needed.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype(numpy.float32) * (1./256)
        x_test  = x_test.astype(numpy.float32) * (1./256)

        self._x_train_base = x_train
        self._y_train_base = y_train        
        self._x_test_base  = x_test
        self._y_test_base  = y_test        
        
        self._base_shape = [28,28]
        
        self._random = random.Random(seed)
        
    def next_train_batch(self, batch_size=10, n_digits=2):
        '''
        Create a new training batch of a specified number of images,
        with the specified number of digits per image.
        
        Parameters
        ----------
        batch_size : int (default = 10)
        n_digits : int (default = 2)
        Returns
        -------
        images : ndarray (shape = [batch_size, 28, n_digits*28]
        labels : ndarray (shape = [batch_size] )

        Examples
        --------
        # Get a batch with 10 images, each a 2 digit number (Default):
        images, labels = generator.next_train_batch()
        
        # Get a batch with 20 images, each a 3 digit number:
        images, labels = generator.next_train_batch(20, 3)
        '''
        
        # First, allocate memory to hold the output data:
        # Data is stored as [B, H, W, C]

        images = numpy.zeros([batch_size, self._base_shape[0], n_digits*self._base_shape[1]])
        labels = numpy.zeros([batch_size], dtype=numpy.int32)
        
        indexes = numpy.asarray(
            self._random.sample(
                range(len(self._x_train_base)),
                batch_size*n_digits
            )
        )

        indexes = indexes.reshape([batch_size, n_digits])
        dims = [10] * n_digits
        for b in range(batch_size):
            # pick a random number from the train set:

            for n in range(n_digits):
                i = indexes[b][n]
                images[b, :, n*28:(n+1)*28] = self._x_train_base[i]
                
            this_label = [ self._y_train_base[j] for j in indexes[b] ]
            labels[b] = numpy.ravel_multi_index(this_label, dims)
                
        return images, labels

                

    def next_test_batch(self, batch_size=10, n_digits=2):
        '''
        Create a new testing batch of a specified number of images,
        with the specified number of digits per image.
        
        Parameters
        ----------
        batch_size : int (default = 10)
        n_digits : int (default = 2)
        Returns
        -------
        images : ndarray (shape = [batch_size, 28, n_digits*28]
        labels : ndarray (shape = [batch_size] )

        Examples
        --------
        # Get a batch with 10 images, each a 2 digit number (Default):
        images, labels = generator.next_train_batch()
        
        # Get a batch with 20 images, each a 3 digit number:
        images, labels = generator.next_train_batch(20, 3)
        '''
        
        # First, allocate memory to hold the output data:
        # Data is stored as [B, H, W, C]

        images = numpy.zeros([batch_size, self._base_shape[0], n_digits*self._base_shape[1]])
        labels = numpy.zeros([batch_size])
        
        indexes = numpy.asarray(
            self._random.sample(
                range(len(self._x_test_base)),
                batch_size*n_digits
            )
        )

        indexes = indexes.reshape([batch_size, n_digits])
        dims = [10] * n_digits
        for b in range(batch_size):
            # pick a random number from the train set:

            for n in range(n_digits):
                i = indexes[b][n]
                images[b, :, n*28:(n+1)*28] = self._x_test_base[i]
                
            this_label = [ self._y_test_base[j] for j in indexes[b] ]
            labels[b] = numpy.ravel_multi_index(this_label, dims)
                
        return images, labels


                
                
                