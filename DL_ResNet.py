#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# In[2]:


import mnist_reader
import matplotlib.pyplot as plt


from tf_utils import random_mini_batches, convert_to_one_hot, predict


# # 2. Loading Test and Train set Features and Labels

# In[3]:


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


# # 3. Normalizing images and Unflattening for Data Preparation

# In[4]:


###NORMALIZING AND CHECKING THE SHAPES OF TRAIN AND TEST SETS



X_train = X_train/255

X_test = X_test/255

print("Feature Train and Test datasets are normalized")


img_size = 28

#print(X_train[1])


X_train = X_train.reshape(X_train.shape[0],img_size,img_size,1)
X_test = X_test.reshape(X_test.shape[0],img_size,img_size,1)
input_size = (img_size, img_size,1)
y_train = (convert_to_one_hot(y_train, 10)).T
y_test = (convert_to_one_hot(y_test, 10)).T

print("Shape of Train set features (X_train) :  ",X_train.shape)
print("Shape of Train set labels (y_train) :  ",y_train.shape)
print("Shape of Test set features (X_test) :  ",X_test.shape)
print("Shape of Test set labels (y_test) :  ",y_test.shape)
#print(X_train[0:783,0])


# # 4. Identity Block for Resnet50

# In[5]:


# Identity block for Conv layers ad window definatitions
def identity_block(X, f, filters, stage, block):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Filters
    F1, F2, F3 = filters 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path 
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
   
    # Third component of main path 
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    # Final step
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    

    return X


# # 5. Convolutional Block for ResNet50

# In[6]:


# The Convolutional block in ResNet50: convolutional_block

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #  Filters
    F1, F2, F3 = filters
    X_shortcut = X

    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path 
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Shortcut Path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


# # 6. Resnet50 Function Architecture 

# In[7]:


# Resnet; CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> 
# CONVBLOCK -> IDBLOCK*3-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER


def ResNet50(input_shape=(28,28,1), classes=10):
    
    # Input as a tensor
    X_input = Input(input_shape)

    # Zero-Padding
    X =  ZeroPadding2D((3, 3))(X_input)
    #print(X)

    # CONV -> BN -> ACTIVATION -> MAXPOOL
    X = Conv2D(64, (7, 7), strides=(3, 3), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # convolutional_block -> 2 identity_blocks 
    X = convolutional_block(X, f=3, filters=[64,64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # convolutional_block -> 3 identity_blocks
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # convolutional_block -> 5 identity_blocks
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # convolutional_block -> 2 identity_blocks
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
    # AVGPOOL 
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Flatten -> FC 
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # model: name='ResNet50'
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model



# In[8]:


model = ResNet50(input_shape=(28,28,1), classes=10)


# # 7. Compiling the Model ( Adam Optimizer and Cross Entropy Loss function)

# In[9]:


# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# # 8. Training the Model (Epochs = 6, Batch Size = 500)

# In[10]:


model.fit(X_train, y_train, epochs = 6, batch_size = 500)


# # 9. Testing the Model

# In[12]:


preds = model.evaluate(X_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


# # 10. Summary of the Model Architecture

# In[13]:


model.summary()


# In[ ]:




