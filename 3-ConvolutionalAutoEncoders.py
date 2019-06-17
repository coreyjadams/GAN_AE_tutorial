#!/usr/bin/env python
# coding: utf-8

# ## Convolutional Auto-Encoders
# 
# This notebook will be relatively brief, as we will implement a different technique for autoencoders.  All of our data flow, training, etc will be identical, except that the model will be different.

# In[1]:


# This is the same from noteboook 2:
# Load the data generator and tensorflow:

import tensorflow as tf
import numpy
tf.enable_eager_execution()
from src.utils import data_generator


# In[2]:


# Note that we are loading different models here:
from src.models import convolutional_AE

# Core training parameters:
N_TRAINING_ITERATION = 5000
BATCH_SIZE = 64
NUM_DIGITS = 1


# In[3]:


# Let's set up our models:

encoder = convolutional_AE.Encoder()
decoder = convolutional_AE.Decoder()



# In[4]:


data_gen = data_generator.mnist_generator()

# Load some data:
batch_images, batch_labels = data_gen.next_train_batch(BATCH_SIZE, NUM_DIGITS)
# Reshape the data:
batch_images = batch_images.reshape([BATCH_SIZE, 28, 28*NUM_DIGITS, 1])

intermediate_state = encoder(batch_images)
decoded_images = decoder(intermediate_state)


# In[5]:


print("Here is the encoder model:")
encoder.summary()
print("Here is the intermediate representation shape:")
print(intermediate_state.shape)
print("Here is the decoder model: ")
decoder.summary()
print("Here is the decoded images shape:")
print(decoded_images.shape)


# This model has many fewer parameters than the dense neural networks.  Let's see how the performance does!

# In[6]:


# For an optimizer, we will use Adam Optimizer:

optimizer = tf.train.AdamOptimizer()



# In[7]:


data_gen = data_generator.mnist_generator()

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
        intermediate_state = encoder(batch_images)
        decoded_images = decoder(intermediate_state)
        loss_value = tf.losses.mean_squared_error(batch_images, decoded_images)

    if i % 50 == 0:
        test_images, test_labels = data_gen.next_test_batch(BATCH_SIZE, NUM_DIGITS)
        test_images = test_images.reshape(
            [BATCH_SIZE, 28, 28*NUM_DIGITS, 1])
        val_intermediate_state = encoder(test_images)
        val_decoded_images = decoder(intermediate_state)
        val_loss_value = tf.losses.mean_squared_error(batch_images, decoded_images)
        val_loss_history.append(val_loss_value.numpy())
        val_steps.append(i)
        
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables


    loss_history.append(loss_value.numpy())

    # Apply the update to the model:
    grads = tape.gradient(loss_value, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars),
                             global_step=tf.train.get_or_create_global_step())

    if i % 50 == 0:
        print("Step {}, loss {}".format(i, loss_history[-1]))


# In[8]:


# from matplotlib import pyplot as plt
# get_ipython().magic(u'matplotlib inline')

# fig = plt.figure(figsize=(16,9))
# plt.plot(range(len(loss_history)), loss_history, label="Train Loss")
# plt.plot(val_steps, val_loss_history, label="Test Loss")
# plt.grid(True)
# plt.legend(fontsize=25)
# plt.xlabel("Training step", fontsize=25)
# plt.show()


# # As in the fully connected network, the loss appears to have converged pretty well.  Let's look at some images on the other side of the encoder:

# # In[9]:


# def run_inference(_encoder, _decoder, input_images):

#     N_INFERENCE_IMAGES = input_images.shape[0]
#     input_images = input_images.reshape(N_INFERENCE_IMAGES, 
#                                         28, 
#                                         28*NUM_DIGITS,
#                                        1)
#     intermediate_rep = _encoder(input_images)
#     decoded_images = _decoder(intermediate_rep)
#     decoded_images = decoded_images.numpy().reshape(
#         N_INFERENCE_IMAGES*28, NUM_DIGITS*28)
    
#     return intermediate_rep, decoded_images


# # In[10]:


# N_INFERENCE_IMAGES = 2


# original_images, labels = data_gen.next_train_batch(
#     N_INFERENCE_IMAGES, NUM_DIGITS)

# intermediate_rep, decoded_images = run_inference(
#     encoder, decoder, original_images)
# original_images = original_images.reshape(N_INFERENCE_IMAGES*28, NUM_DIGITS*28)


# # In[11]:


# fig = plt.figure(figsize=(5, N_INFERENCE_IMAGES*5))
# plt.imshow(original_images)
# plt.show()
# fig = plt.figure(figsize=(5, N_INFERENCE_IMAGES*5))
# plt.imshow(decoded_images)
# plt.show()


# # Again, the decoded images are recognizable and correspond to the input images, though they are not identical.

# # In[12]:


# plt.imshow(intermediate_rep)
# plt.show()


# # In[13]:


# N_INFERENCE_IMAGES = 2


# original_images, labels = data_gen.next_test_batch(
#     N_INFERENCE_IMAGES, NUM_DIGITS)

# intermediate_rep, decoded_images = run_inference(
#     encoder, decoder, original_images)
# original_images = original_images.reshape(N_INFERENCE_IMAGES*28, NUM_DIGITS*28)

# fig = plt.figure(figsize=(5, N_INFERENCE_IMAGES*5))
# plt.imshow(original_images)
# plt.show()
# fig = plt.figure(figsize=(5, N_INFERENCE_IMAGES*5))
# plt.imshow(decoded_images)
# plt.show()


# Once again, the network is performing well on the validation test set.  Save the networks since they will be loaded and used in the next notebook:

# In[14]:


# Save the trained models again:
encoder.save_weights("saved_models/pretrained/conv_encoder.h5")
decoder.save_weights("saved_models/pretrained/conv_decoder.h5")


# In[ ]:




