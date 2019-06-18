#!/usr/bin/env python
# coding: utf-8

# ## Comparing Auto Encoders
# 
# In the previous tutorials, we trained two autoencoders to compress hand written digits into a single, 10-long floating point number per digit.  The first network was a standard fully connected neural network, and the second was a much smaller convolutional neural network - about 10x smaller in trainable parameters.
# 
# Which was better?  It's hard to answer that question just from looking at a few output images, and certainly examining the intermediate representation by eye is not super useful.  In this notebook we'll try two techniques to measure just "how good" the autoencoder/decoder pairs are doing:
# 
# 1) Comparision of Mean Squared Error on validation set
# 
# 2) Comparision of performance of classification network on decoded images
# 
# 

# In[1]:


# This is the same from noteboook 2:
# Load the data generator and tensorflow:

import tensorflow as tf
import numpy
tf.enable_eager_execution()
from src.utils import data_generator

from src.models import convolutional_AE
from src.models import neural_net_AE




# In[2]:


data_gen = data_generator.mnist_generator()

fc_encoder   = neural_net_AE.Encoder()
conv_encoder = convolutional_AE.Encoder()

fc_decoder   = neural_net_AE.Decoder(1)
conv_decoder = convolutional_AE.Decoder()



# Like in the other notebooks, the models are not really initialized until you go through them once with input data.  So we will run through once for each model to make sure they are initialized, then we can load them:

# In[3]:


# Restore the trained models for each of the networks:
BATCH_SIZE=1
NUM_DIGITS=1

data_gen = data_generator.mnist_generator()

# Load some data:
batch_images, batch_labels = data_gen.next_train_batch(BATCH_SIZE, NUM_DIGITS)
# Reshape the data:
fc_batch_images = batch_images.reshape([BATCH_SIZE, 28*28*NUM_DIGITS])

fc_intermediate_state = fc_encoder(fc_batch_images)
fc_decoded_images = fc_decoder(fc_intermediate_state)

conv_batch_images = batch_images.reshape([BATCH_SIZE, 28, 28*NUM_DIGITS, 1])


conv_intermediate_state = conv_encoder(conv_batch_images)
conv_decoded_images = conv_decoder(conv_intermediate_state)



fc_encoder.load_weights("saved_models/pretrained/nn_encoder.h5")
fc_decoder.load_weights("saved_models/pretrained/nn_decoder.h5")
conv_encoder.load_weights("saved_models/pretrained/conv_encoder.h5")
conv_decoder.load_weights("saved_models/pretrained/conv_decoder.h5")


# In[12]:


# We will skip the data generator since this is single image autoencoding.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float32) * (1./256)
x_test  = x_test.astype(numpy.float32) * (1./256)




fc_x_test = x_test.reshape(10000, 28*28).astype(numpy.float32)
conv_x_test = x_test.reshape(10000, 28, 28, 1).astype(numpy.float32)


# In[ ]:





# In[5]:


# fc_intermediate_state = fc_encoder(fc_x_test)
# fc_decoded = fc_decoder(fc_intermediate_state)

# conv_intermediate_state = conv_encoder(conv_x_test)
# conv_decoded = conv_decoder(conv_intermediate_state)


# In[6]:


# fc_loss = tf.losses.mean_squared_error(fc_x_test, fc_decoded)
# conv_loss = tf.losses.mean_squared_error(conv_x_test, conv_decoded)


# In[7]:


# print("Fully connected test loss: ", fc_loss.numpy())
# print("Convolutional test loss: ", conv_loss.numpy())


# Based on this, the convolutional auto encoder is marginally better than the fully connected:  it has a slightly lower loss on the test set.

# ### Classifying decoded data
# 
# Let's look at another comparison.  We'll quickly spin up a classifier for mnist digits and then see how well the decoded data is doing at recreating it's original state by classifying the output of the decoder.
# 
# There is an mnist model ready to run in the models folder, so let's load and use that:

# In[8]:


from src.models import mnist_classifier

model = mnist_classifier.mnist_classifier(n_output=10)


# In[9]:


N_TRAINING_ITERATION = 5000
BATCH_SIZE = 64
data_gen = data_generator.mnist_generator()

optimizer = tf.train.AdamOptimizer()

loss_history = []
val_loss_history = []
val_steps = []

for i in range(N_TRAINING_ITERATION):

    # Load some data:
    batch_images, batch_labels = data_gen.next_train_batch(BATCH_SIZE, NUM_DIGITS)
    # Reshape the data:
    batch_images = batch_images.reshape(
        [BATCH_SIZE, 28, 28*NUM_DIGITS, 1])

    with tf.GradientTape() as tape:
        logits = model(batch_images)
        loss_value = tf.losses.sparse_softmax_cross_entropy(batch_labels, logits)

        
    trainable_vars = model.trainable_variables


    loss_history.append(loss_value.numpy())

    # Apply the update to the model:
    grads = tape.gradient(loss_value, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars),
                             global_step=tf.train.get_or_create_global_step())

    if i % 50 == 0:
        print("Step {}, loss {}".format(i, loss_history[-1]))


# In[20]:


val_images, val_labels = data_gen.next_test_batch(512, 1)
val_images = val_images.reshape(512, 28, 28 ,1)
logits = model(val_images)


val_loss = tf.losses.sparse_softmax_cross_entropy(val_labels, logits)

print(val_loss)

# Get the validation accuracy:


# In[ ]:
model.save_weights("saved_models/pretrained/mnist_classifier.h5")




